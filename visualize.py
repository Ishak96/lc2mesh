#!/usr/bin/env python3
"""Visualize a saved reconstruction: lightcurve fit, voxel measures, rotating video.

This is the script form of the reference ``visualize.ipynb``. It reads a mesh
from ``recon/reconstructed_ast<ID>.stl`` (written by ``run.py``), re-evaluates it
against the observed lightcurves with the convex forward model, prints the
official voxel measures against the true STL when it is available, and renders a
rotating mp4 into ``videos/``.

Usage:
    python visualize.py --asteroid 3 [--gpu 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

from lc2mesh import config as cfg  # noqa: E402
from lc2mesh.constants import ASTEROID_IDS, OBSERVERS  # noqa: E402
from lc2mesh.data import get_model_folder, load_data  # noqa: E402
from lc2mesh.eval import relative_volume_difference_voxelized  # noqa: E402
from lc2mesh.forward_model import AsteroidForwardModel, observers_to_unit_tensor  # noqa: E402
from lc2mesh.mesh import normalize_mesh_to_challenge_cylinder  # noqa: E402
from lc2mesh.metrics import compute_per_camera_metrics  # noqa: E402
from lc2mesh.pipeline import facet_geometry  # noqa: E402
from lc2mesh.prior import create_ellipsoidal_prior_mesh  # noqa: E402
from lc2mesh.utils import set_seed, setup_device  # noqa: E402
from lc2mesh.visualization import (  # noqa: E402
    plot_camera_grid,
    plot_interactive_mesh,
    plot_lightcurve_comparison,
    render_rotating_mesh_video,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# The viewer reproduces the notebook settings, which differ from training on
# purpose: float64 arithmetic and the plain convex forward model.
VIEW_LAMBERT_WEIGHT = 0.3
VIEW_SEED = 42
RECON_COLOR = "#c0b08a"
TARGET_COLOR = "#a0c0ff"
N_FRAMES = 60  # 6 s at 10 fps
FPS = 10
ELEVATION = 20.0
BACKGROUND_COLOR = "white"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a reconstructed asteroid mesh and its lightcurve fit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--asteroid", type=int, required=True, choices=ASTEROID_IDS)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (CPU if unavailable).")
    parser.add_argument(
        "--prior-shape",
        default=cfg.PRIOR_SHAPE_TYPE,
        choices=("ellipsoid", "peanut", "cube", "diamond", "irregular"),
        help="Prior archetype, only used to look up the cylinder radius.",
    )
    parser.add_argument("--recon-dir", type=Path, default=PROJECT_ROOT / "recon")
    parser.add_argument("--video-dir", type=Path, default=PROJECT_ROOT / "videos")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Where to write the lightcurve figures (default: results/ast<ID>).",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip the rotating mp4.")
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also write interactive plotly HTML views of the meshes.",
    )
    return parser.parse_args(argv)


def predicted_lightcurves_from_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    phases: torch.Tensor,
    e_sun: torch.Tensor,
    e_obs: torch.Tensor,
) -> torch.Tensor:
    normals, areas = facet_geometry(vertices, faces)
    fm = AsteroidForwardModel(normals=normals, lambert_weight=VIEW_LAMBERT_WEIGHT)
    A = fm.build_forward_matrix(phases=phases, e_sun=e_sun, e_obs=e_obs)
    # AsteroidForwardModel computes in float32; match it so the einsum dtypes agree.
    raw = torch.einsum("cpf,f->cp", A.to(areas.dtype), areas)
    return raw / raw.mean(dim=1, keepdim=True).clamp_min(1e-12)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asteroid_id = int(args.asteroid)
    figure_dir = args.figure_dir or (PROJECT_ROOT / "results" / f"ast{asteroid_id}")
    figure_dir.mkdir(parents=True, exist_ok=True)

    torch.set_default_dtype(torch.float64)
    set_seed(VIEW_SEED)
    device = setup_device(args.gpu, allow_cpu=True)

    recon_stl_file = Path(args.recon_dir) / f"reconstructed_ast{asteroid_id}.stl"
    if not recon_stl_file.exists():
        raise FileNotFoundError(
            f"Reconstruction STL not found: {recon_stl_file.resolve()}. "
            f"Run ./run.sh <GPU_ID> {asteroid_id} first."
        )

    camera_names, e_obs = observers_to_unit_tensor(
        OBSERVERS, device=device, dtype=torch.float64
    )
    e_sun = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64, device=device)

    ds_real = load_data(asteroid_id, origin="real", curve_type=cfg.CURVE_TYPE)
    observed = torch.as_tensor(
        ds_real.data[camera_names].to_numpy(dtype=np.float64, copy=True).T,
        dtype=torch.float64,
        device=device,
    )
    observed_norm = observed / observed.mean(dim=1, keepdim=True).clamp_min(1e-12)
    n_phase = observed.shape[1]
    phases = torch.arange(n_phase, dtype=torch.float64, device=device) * (
        2.0 * torch.pi / n_phase
    )

    prior = create_ellipsoidal_prior_mesh(
        asteroid_id=asteroid_id,
        subdivisions=cfg.N_SUBDIVISIONS,
        device=device,
        shape_type=args.prior_shape,
    )
    R_CYL = float(prior["R_CYLINDER"])

    recon_mesh = normalize_mesh_to_challenge_cylinder(
        trimesh.load_mesh(recon_stl_file), R_CYL
    )
    recon_vertices = torch.as_tensor(recon_mesh.vertices, dtype=torch.float64, device=device)
    recon_faces = torch.as_tensor(recon_mesh.faces, dtype=torch.long, device=device)

    gt_stl_file = get_model_folder(asteroid_id) / f"asteroid{asteroid_id}.stl"
    true_mesh = None
    if gt_stl_file.exists():
        true_mesh = normalize_mesh_to_challenge_cylinder(trimesh.load_mesh(gt_stl_file), R_CYL)
    else:
        print(f"Warning: GT STL file not found at {gt_stl_file}")

    print(f"Asteroid {asteroid_id}: cameras={len(camera_names)}, phases={n_phase}")
    print(f"Reconstruction STL: {recon_stl_file.resolve()}")
    print(f"Cylinder radius: {R_CYL:.3f}")

    pred_norm = predicted_lightcurves_from_mesh(
        recon_vertices, recon_faces, phases, e_sun, e_obs
    )
    per_camera_df = compute_per_camera_metrics(observed_norm, pred_norm, camera_names)

    print(
        f"Reconstruction extents: "
        f"x[{recon_mesh.vertices[:, 0].min():.3f}, {recon_mesh.vertices[:, 0].max():.3f}] "
        f"y[{recon_mesh.vertices[:, 1].min():.3f}, {recon_mesh.vertices[:, 1].max():.3f}] "
        f"z[{recon_mesh.vertices[:, 2].min():.3f}, {recon_mesh.vertices[:, 2].max():.3f}] "
        f"(cylinder bound: R={R_CYL}, z in [-1, 1])"
    )

    if true_mesh is not None:
        print(
            f"GT extents: "
            f"x[{true_mesh.vertices[:, 0].min():.3f}, {true_mesh.vertices[:, 0].max():.3f}] "
            f"y[{true_mesh.vertices[:, 1].min():.3f}, {true_mesh.vertices[:, 1].max():.3f}] "
            f"z[{true_mesh.vertices[:, 2].min():.3f}, {true_mesh.vertices[:, 2].max():.3f}]"
        )
        pred_mesh = trimesh.Trimesh(
            vertices=recon_mesh.vertices, faces=recon_mesh.faces, process=False
        )
        measure1, measure2 = relative_volume_difference_voxelized(
            pred_mesh, true_mesh, pitch=cfg.VOXEL_PITCH
        )
        print(
            f"Official voxel measures vs GT (pitch={cfg.VOXEL_PITCH}); "
            "0 = identical, lower is better:"
        )
        print(f"  measure1 (1 - IoU)={measure1:.6f} | measure2 (sym-diff/sum)={measure2:.6f}")
    else:
        print("GT mesh not available -> skipping voxel measure calculation.")

    if args.html:
        fig_mesh = plot_interactive_mesh(
            vertices=recon_mesh.vertices,
            faces=recon_mesh.faces,
            title=(
                f"Asteroid {asteroid_id}: reconstructed mesh "
                f"({recon_mesh.faces.shape[0]} faces) | "
                f"mean_r={per_camera_df['Pearson_R'].mean():.3f}"
            ),
            mesh_color=RECON_COLOR,
            show_edges=False,
        )
        html_path = figure_dir / f"reconstruction_ast{asteroid_id}.html"
        fig_mesh.write_html(str(html_path))
        print(f"Saved interactive mesh: {html_path}")
        if true_mesh is not None:
            gt_fig_mesh = plot_interactive_mesh(
                vertices=true_mesh.vertices,
                faces=true_mesh.faces,
                title=(
                    f"Asteroid {asteroid_id}: GT mesh, cylinder-normalized "
                    f"({true_mesh.faces.shape[0]} faces)"
                ),
                mesh_color=TARGET_COLOR,
                show_edges=False,
            )
            gt_html_path = figure_dir / f"true_ast{asteroid_id}.html"
            gt_fig_mesh.write_html(str(gt_html_path))
            print(f"Saved interactive GT mesh: {gt_html_path}")

    print(
        f"Final lightcurve fit: mean_r={per_camera_df['Pearson_R'].mean():.4f}, "
        f"min_r={per_camera_df['Pearson_R'].min():.4f}, "
        f"mean MSE={per_camera_df['MSE'].mean():.6f}"
    )
    print(per_camera_df.sort_values("Pearson_R").to_string())

    target_cam = cfg.TARGET_CAM if cfg.TARGET_CAM in camera_names else camera_names[0]
    fig, _ = plot_lightcurve_comparison(
        obs_norm=observed_norm.detach().cpu().numpy(),
        pred_norm=pred_norm.detach().cpu().numpy(),
        camera_keys=camera_names,
        target_cam_id=target_cam,
    )
    fig.savefig(figure_dir / "visualize_lightcurve_comparison.png", dpi=150)
    plt.close(fig)

    phase_deg = np.rad2deg(phases.detach().cpu().numpy())
    fig, _ = plot_camera_grid(observed_norm, pred_norm, camera_names, phase_axis=phase_deg)
    fig.savefig(figure_dir / "visualize_camera_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved lightcurve figures: {figure_dir}")

    if not args.no_video:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        panels = [("Reconstruction", recon_mesh, RECON_COLOR)]
        if true_mesh is not None:
            panels = [
                ("Target", true_mesh, TARGET_COLOR),
                ("Reconstruction", recon_mesh, RECON_COLOR),
            ]
        suffix = "true_vs_optimized" if len(panels) == 2 else "reconstruction"
        video_path = video_dir / f"{suffix}_ast{asteroid_id}.mp4"
        render_rotating_mesh_video(
            panels,
            video_path,
            title_prefix=f"Asteroid {asteroid_id}: ",
            limit=max(R_CYL, 1.0) * 1.05,
            n_frames=N_FRAMES,
            fps=FPS,
            elevation=ELEVATION,
            background_color=BACKGROUND_COLOR,
        )
        print(f"Saved rotating video ({N_FRAMES} frames @ {FPS} fps): {video_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
