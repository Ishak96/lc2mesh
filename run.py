#!/usr/bin/env python3
"""End-to-end INR + graph-CNN asteroid reconstruction from lightcurves.

This script is a direct transcription of the reference notebook
(``inr_forward_train.ipynb``): same configuration, same two-stage optimization,
same convex-floor acceptance gate, same evaluation. It

1. loads the observed lightcurves and the convex-inversion prior,
2. trains Stage 1 (coarse INR, convex forward model),
3. refines with Stage 2 (graph CNN, self-occlusion forward model) and applies
   the convex-floor acceptance gate,
4. evaluates the final mesh against the observed lightcurves,
5. computes the official voxel measures against the true STL when available,
6. saves the reconstructed mesh, the checkpoint, the metrics and the figures.

Usage:
    python run.py --gpu 0 --asteroid 3
"""

from __future__ import annotations

import os

# cuBLAS workspace must be configured before any CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless rendering; figures are written to disk

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

from lc2mesh import config as cfg  # noqa: E402
from lc2mesh.constants import ASTEROID_IDS, OBSERVERS  # noqa: E402
from lc2mesh.data import get_model_folder, load_data  # noqa: E402
from lc2mesh.eval import relative_volume_difference_voxelized  # noqa: E402
from lc2mesh.forward_model import observers_to_unit_tensor  # noqa: E402
from lc2mesh.mesh import normalize_mesh_to_challenge_cylinder  # noqa: E402
from lc2mesh.metrics import (  # noqa: E402
    compute_mse,
    compute_per_camera_metrics,
    mesh_reconstruction_metrics,
)
from lc2mesh.model import LightcurveToMesh  # noqa: E402
from lc2mesh.pipeline import LightcurvePipeline  # noqa: E402
from lc2mesh.prior import create_ellipsoidal_prior_mesh  # noqa: E402
from lc2mesh.training import (  # noqa: E402
    TrainingConfig,
    split_camera_indices,
    train_stage,
)
from lc2mesh.utils import (  # noqa: E402
    sample_convex_hull_surface,
    set_seed,
    setup_device,
    vertex_normals_from_faces,
)
from lc2mesh.visualization import plot_camera_grid, plot_lightcurve_comparison  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct an asteroid mesh from its lightcurves (INR + GCN).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="CUDA device index to train on (falls back to CPU if unavailable).",
    )
    parser.add_argument(
        "--asteroid",
        type=int,
        required=True,
        choices=ASTEROID_IDS,
        help="Challenge asteroid id (1-10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Root directory for ckpt/, recon/ and results/ outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asteroid_id = int(args.asteroid)
    output_root = Path(args.output_dir).resolve()

    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)

    # ---------------------------------------------------------------- paths --
    convex_inversion_dir = PROJECT_ROOT / "convex_inversions"
    checkpoint_dir = output_root / "ckpt"
    recon_dir = output_root / "recon"
    results_dir = output_root / "results" / f"ast{asteroid_id}"
    for directory in (checkpoint_dir, recon_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    reference_stl_path_eval = get_model_folder(asteroid_id) / f"asteroid{asteroid_id}.stl"
    if cfg.REFERENCE_MESH_SOURCE == "true":
        if not cfg.ALLOW_ORACLE_REFERENCE:
            raise ValueError(
                "True-mesh training is an oracle experiment; set ALLOW_ORACLE_REFERENCE=True explicitly."
            )
        reference_stl_path = reference_stl_path_eval
        reference_mesh_label = "ORACLE true STL"
    elif cfg.REFERENCE_MESH_SOURCE == "convex_inversion":
        reference_stl_path = (
            convex_inversion_dir / f"asteroid{asteroid_id}_simulated_intensity.stl"
        )
        reference_mesh_label = "external simulated-lightcurve convex inversion"
    else:
        raise ValueError("REFERENCE_MESH_SOURCE must be 'true' or 'convex_inversion'")

    checkpoint_path = checkpoint_dir / f"inr_forward_train_{asteroid_id}.pth"

    set_seed(cfg.SEED)
    device = setup_device(args.gpu)
    print(f"Training reference: {reference_mesh_label} -> {reference_stl_path}")
    print(
        "Validation is conditional on the external prior, not a clean end-to-end held-out test."
    )

    # ------------------------------------------------------- data and model --
    camera_names, e_obs = observers_to_unit_tensor(
        OBSERVERS, device=device, dtype=torch.float32
    )
    e_sun = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    ds_real = load_data(asteroid_id, origin="real", curve_type=cfg.CURVE_TYPE)
    observed = torch.as_tensor(
        ds_real.data[camera_names].to_numpy(dtype=np.float32, copy=True).T,
        dtype=torch.float32,
        device=device,
    )
    observed_norm = observed / observed.mean(dim=1, keepdim=True).clamp_min(1e-12)
    n_phase = observed.shape[1]
    phases = torch.arange(n_phase, dtype=torch.float32, device=device) * (
        2.0 * torch.pi / n_phase
    )
    train_indices, validation_indices = split_camera_indices(
        len(camera_names),
        seed=cfg.SPLIT_SEED,
        validation_fraction=cfg.VALIDATION_FRACTION,
        device=device,
    )

    prior = create_ellipsoidal_prior_mesh(
        asteroid_id=asteroid_id,
        subdivisions=cfg.N_SUBDIVISIONS,
        device=device,
        shape_type=cfg.PRIOR_SHAPE_TYPE,
    )
    base_vertices = prior["prior_vertices_batch"].to(dtype=torch.float32)
    faces = prior["prior_faces_t"]
    n_vertices = int(base_vertices.shape[1])
    base_normals = vertex_normals_from_faces(base_vertices, faces)
    model_config = dict(
        inr_hidden_dim=cfg.INR_HIDDEN_DIM,
        inr_num_layers=cfg.INR_NUM_LAYERS,
        coarse_w0=cfg.COARSE_W0,
        inr_final_init_scale=cfg.INR_FINAL_INIT_SCALE,
        gcn_architecture=cfg.GCN_ARCHITECTURE,
        gcn_hidden_dim=cfg.GCN_HIDDEN_DIM,
        gcn_num_layers=cfg.GCN_NUM_LAYERS,
        gcn_final_init_scale=cfg.GCN_FINAL_INIT_SCALE,
        canonicalize_coarse=True,
        R=prior["R_CYLINDER"],
    )
    net = LightcurveToMesh(faces=faces, **model_config).to(device)
    n_params_inr = sum(parameter.numel() for parameter in net.inr.parameters())
    n_params_gcn = sum(parameter.numel() for parameter in net.gcn.parameters())
    print(f"Asteroid {asteroid_id}: cameras={len(camera_names)}, phases={n_phase}")
    print(f"Training cameras: {[camera_names[index] for index in train_indices.tolist()]}")
    print(
        f"Validation cameras: {[camera_names[index] for index in validation_indices.tolist()]}"
    )
    print(f"Prior: {n_vertices} vertices, {len(faces)} faces, cylinder R={prior['R_CYLINDER']}")
    print(f"Parameters: INR={n_params_inr:,} | GCN={n_params_gcn:,}")

    # ------------------------------------- differentiable lightcurve pipeline --
    pipeline = LightcurvePipeline(
        faces=faces,
        phases=phases,
        e_sun=e_sun,
        e_obs=e_obs,
        device=device,
        lambert_weight=cfg.LAMBERT_WEIGHT,
        occlusion_enabled=cfg.OCCLUSION_ENABLED,
        occlusion_refresh_every=cfg.OCCLUSION_REFRESH_EVERY,
        occlusion_n_phases=cfg.OCCLUSION_N_PHASES,
        occlusion_eps=cfg.OCCLUSION_EPS,
    )

    # Sanity check: refined lightcurve gradients reach both stages.
    _, refined0 = net(base_vertices, base_normals)
    pred0 = pipeline.predicted_lightcurves(refined0[0])
    loss0 = compute_mse(observed_norm, pred0)
    loss0.backward()
    grad_inr = torch.sqrt(
        sum((p.grad**2).sum() for p in net.inr.parameters() if p.grad is not None)
    )
    grad_gcn = torch.sqrt(
        sum((p.grad**2).sum() for p in net.gcn.parameters() if p.grad is not None)
    )
    all_finite = all(
        torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None
    )
    print(
        f"Sanity check: loss={float(loss0.item()):.6f} | grad_norm INR={float(grad_inr):.3e}, "
        f"GCN={float(grad_gcn):.3e} | all_finite={bool(all_finite)}"
    )
    net.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------------------------------- convex-hull prior / containment --
    if not reference_stl_path.is_file():
        raise FileNotFoundError(f"Explicit geometric prior required: {reference_stl_path}")
    reference_mesh_norm = normalize_mesh_to_challenge_cylinder(
        trimesh.load_mesh(reference_stl_path),
        prior["R_CYLINDER"],
    )
    hull_samples = sample_convex_hull_surface(
        np.asarray(reference_mesh_norm.vertices),
        n_points=cfg.HULL_N_SAMPLES,
        seed=cfg.SEED,
    )
    hull_points = torch.as_tensor(
        hull_samples, dtype=torch.float32, device=device
    ).unsqueeze(0)
    # Half-space planes of the convex hull (outward normals n, offsets d): a point x
    # is inside iff n.x <= d for every face. Stage 2 uses these to forbid outward
    # bulging while allowing inward carving (the true shape is inside the hull).
    _hull = reference_mesh_norm.convex_hull
    _hull_normals = np.asarray(_hull.face_normals, dtype=np.float32)
    _hull_offsets = (
        _hull_normals * np.asarray(_hull.triangles, dtype=np.float32)[:, 0]
    ).sum(axis=1)
    containment_planes = (
        torch.as_tensor(_hull_normals, dtype=torch.float32, device=device),
        torch.as_tensor(_hull_offsets, dtype=torch.float32, device=device),
    )
    print(
        f"{reference_mesh_label}: {hull_points.shape[1]} convex-hull samples, "
        f"{len(_hull_offsets)} containment planes"
    )

    # ------------------------------------------------------------- Stage 1 ----
    def report_progress(row):
        if row["step"] == 0 or row["step"] % cfg.PRINT_EVERY == 0:
            print(
                f"step {row['step']:4d} | train MSE={row['train_mse']:.6f} | "
                f"validation MSE={row['validation_mse']:.6f}"
            )

    config1 = TrainingConfig(
        steps=cfg.STAGE1_STEPS,
        lr_inr=cfg.LR_INR_STAGE1,
        hull_weight=cfg.LAMBDA_CHAMFER,
        laplacian_weight=cfg.LAMBDA_LAPLACIAN,
        eval_every=cfg.EVAL_EVERY,
        scheduler_patience=cfg.SCHEDULER_PATIENCE,
        selection=cfg.SELECTION_STAGE1,
    )
    pipeline.state["active"] = False  # Stage 1: convex forward model (convex-hull prior)
    t_start = time.perf_counter()
    stage1_result = train_stage(
        net,
        base_vertices,
        base_normals,
        pipeline.predicted_lightcurves,
        observed_norm,
        hull_points,
        train_indices,
        validation_indices,
        config1,
        stage=1,
        progress=report_progress,
    )
    with torch.no_grad():
        verts_s1, _ = net(base_vertices, base_normals)
        verts_s1 = verts_s1.detach().clone()
    best1_eval = next(
        row for row in stage1_result.evaluations if row["step"] == stage1_result.best_step
    )
    print(
        f"Stage 1: {(time.perf_counter() - t_start) / 60:.1f} min; selected by "
        f"{cfg.SELECTION_STAGE1} at step {stage1_result.best_step}; "
        f"train MSE={best1_eval['train_mse']:.6f}, validation MSE={best1_eval['validation_mse']:.6f}"
    )
    verts_s1_np = verts_s1[0].detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    # ------------------------------------------------------------- Stage 2 ----
    @torch.no_grad()
    def lightcurve_fit(vertices: torch.Tensor, *, occlusion: bool) -> dict[str, float]:
        """Train/validation MSE of a [V, 3] mesh under the convex model
        (occlusion=False) or self-occlusion physics with a FRESHLY ray-cast mask.
        Leaves the occlusion state untouched afterwards."""
        saved = dict(pipeline.state)
        pipeline.state["active"] = bool(occlusion and cfg.OCCLUSION_ENABLED)
        pipeline.state["mask"] = None
        pipeline.state["calls"] = 0
        try:
            prediction = pipeline.predicted_lightcurves(vertices)
        finally:
            pipeline.state.update(saved)
        train = float(compute_mse(observed_norm[train_indices], prediction[train_indices]))
        validation = float(
            compute_mse(
                observed_norm[validation_indices], prediction[validation_indices]
            )
        )
        # 'mean' = equal-weight average of the two split MSEs (not the all-camera MSE).
        return {"train": train, "validation": validation, "mean": 0.5 * (train + validation)}

    # Convex floor: the Stage-1 mesh scored under the convex physics it was fitted
    # with. Stage 2 must beat this (fresh occlusion mask) or it is rejected below.
    stage1_state = {
        name: value.detach().cpu().clone() for name, value in net.state_dict().items()
    }
    convex_floor = lightcurve_fit(verts_s1[0], occlusion=False)
    print(
        f"Convex floor (Stage-1 mesh): train MSE={convex_floor['train']:.6f}, "
        f"validation MSE={convex_floor['validation']:.6f}"
    )

    config2 = TrainingConfig(
        steps=cfg.STAGE2_STEPS,
        lr_inr=cfg.LR_INR_STAGE2,
        lr_graph=cfg.LR_GRAPH,
        hull_weight=cfg.LAMBDA_CHAMFER_STAGE2,
        laplacian_weight=cfg.LAMBDA_LAPLACIAN_STAGE2,
        displacement_weight=cfg.LAMBDA_DISPLACEMENT_STAGE2,
        edge_weight=cfg.LAMBDA_EDGE_STAGE2,
        containment_weight=cfg.LAMBDA_CONTAINMENT_STAGE2,
        eval_every=cfg.EVAL_EVERY,
        scheduler_patience=cfg.SCHEDULER_PATIENCE,
        selection=cfg.SELECTION_STAGE2,
    )
    # Stage 2: enable the self-occlusion forward model so concavities are observable.
    pipeline.reset_occlusion(cfg.OCCLUSION_ENABLED)
    t_start = time.perf_counter()
    stage2_result = train_stage(
        net,
        base_vertices,
        base_normals,
        pipeline.predicted_lightcurves,
        observed_norm,
        hull_points,
        train_indices,
        validation_indices,
        config2,
        stage=2,
        progress=report_progress,
        containment_planes=containment_planes,
    )
    best_evaluation = next(
        row for row in stage2_result.evaluations if row["step"] == stage2_result.best_step
    )
    print(
        f"Stage 2: {(time.perf_counter() - t_start) / 60:.1f} min; selected by "
        f"{cfg.SELECTION_STAGE2} fit at step {stage2_result.best_step}; "
        f"train MSE={best_evaluation['train_mse']:.6f}, "
        f"validation MSE={best_evaluation['validation_mse']:.6f}"
    )
    if best_evaluation["validation_mse"] > stage2_result.evaluations[0]["validation_mse"]:
        print(
            "Note: held-out validation MSE rose during Stage 2 -> the refinement is "
            "overfitting the observed cameras."
        )

    # --- Convex-floor acceptance gate ---
    with torch.no_grad():
        _, verts_s2 = net(base_vertices, base_normals)
        verts_s2 = verts_s2.detach().clone()
    stage2_fit = lightcurve_fit(verts_s2[0], occlusion=True)
    stage1_under_occlusion = lightcurve_fit(verts_s1[0], occlusion=True)
    stage2_accepted = bool(
        stage2_fit[cfg.STAGE2_ACCEPTANCE] < convex_floor[cfg.STAGE2_ACCEPTANCE]
    )
    print(
        f"Gate on {cfg.STAGE2_ACCEPTANCE} cameras (fresh mask): "
        f"Stage-2 MSE={stage2_fit[cfg.STAGE2_ACCEPTANCE]:.6f} vs convex "
        f"floor={convex_floor[cfg.STAGE2_ACCEPTANCE]:.6f} -> "
        f"{'ACCEPTED (refined mesh kept)' if stage2_accepted else 'REJECTED (Stage-1 state restored)'}"
    )
    print(
        f"  Stage-2 mesh, occlusion physics: train={stage2_fit['train']:.6f}, "
        f"validation={stage2_fit['validation']:.6f} | "
        f"Stage-1 mesh, occlusion physics: train={stage1_under_occlusion['train']:.6f}, "
        f"validation={stage1_under_occlusion['validation']:.6f}"
    )
    if stage2_accepted:
        final_mesh_label = "refined"
    else:
        final_mesh_label = "stage1"
        net.load_state_dict(stage1_state)
    final_fit = stage2_fit if stage2_accepted else convex_floor

    best_state = {
        name: value.detach().cpu().clone() for name, value in net.state_dict().items()
    }
    best2 = {
        "step": stage2_result.best_step if stage2_accepted else stage1_result.best_step,
        "loss": final_fit["train"],
        "state": best_state,
    }
    torch.save(
        {
            "state_dict": best_state,
            "model_state": best_state,
            "faces": faces.cpu(),
            "base_vertices": base_vertices.cpu(),
            "config": {
                **model_config,
                "asteroid_id": asteroid_id,
                "n_subdivisions": cfg.N_SUBDIVISIONS,
                "seed": cfg.SEED,
                "split_seed": cfg.SPLIT_SEED,
                "prior_shape_type": cfg.PRIOR_SHAPE_TYPE,
                "lambert_weight": cfg.LAMBERT_WEIGHT,
                "lambda_chamfer": cfg.LAMBDA_CHAMFER,
                "reference_mesh_source": cfg.REFERENCE_MESH_SOURCE,
                "reference_stl_path": str(reference_stl_path),
                "oracle_reference": cfg.REFERENCE_MESH_SOURCE == "true",
                "validation_scope": "conditional on external prior; not a clean end-to-end holdout",
                "occlusion_enabled": cfg.OCCLUSION_ENABLED,
                "occlusion_refresh_every": cfg.OCCLUSION_REFRESH_EVERY,
                "occlusion_n_phases": cfg.OCCLUSION_N_PHASES,
                "lambda_containment_stage2": cfg.LAMBDA_CONTAINMENT_STAGE2,
                "stage2_acceptance": cfg.STAGE2_ACCEPTANCE,
                "train_cameras": [camera_names[index] for index in train_indices.tolist()],
                "validation_cameras": [
                    camera_names[index] for index in validation_indices.tolist()
                ],
                "lr_inr_stage1": cfg.LR_INR_STAGE1,
                "lr_inr_stage2": cfg.LR_INR_STAGE2,
                "lr_graph": cfg.LR_GRAPH,
                "stage1_steps": cfg.STAGE1_STEPS,
                "stage2_steps": cfg.STAGE2_STEPS,
                "training_config_stage1": asdict(config1),
                "training_config_stage2": asdict(config2),
            },
            "best_step": best2["step"],
            "best_loss": best2["loss"],
            "best_train_mse": final_fit["train"],
            "best_validation_mse": final_fit["validation"],
            "selection_metric": cfg.SELECTION_STAGE2,
            "stage2_accepted": stage2_accepted,
            "final_mesh_label": final_mesh_label,
            "convex_floor": convex_floor,
            "stage2_fit": stage2_fit,
            "stage1_under_occlusion": stage1_under_occlusion,
            "stage1": asdict(stage1_result),
            "stage2": asdict(stage2_result),
        },
        checkpoint_path,
    )
    print(f"Final mesh: {final_mesh_label} | Checkpoint: {checkpoint_path}")

    # -------------------------------------------------------- training curves --
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for label, result, offset, color in (
        ("Stage 1", stage1_result, 0, "#d95f02"),
        ("Stage 2", stage2_result, cfg.STAGE1_STEPS, "#1b9e77"),
    ):
        steps = [offset + step for step in result.history["step"]]
        axes[0].plot(steps, result.history["loss"], label=label, color=color)
        eval_steps = [offset + row["step"] for row in result.evaluations]
        axes[1].plot(
            eval_steps,
            [row["train_mse"] for row in result.evaluations],
            label=f"{label} train",
            color=color,
        )
        axes[1].plot(
            eval_steps,
            [row["validation_mse"] for row in result.evaluations],
            label=f"{label} validation",
            color=color,
            ls="--",
        )
        axes[2].plot(steps, result.history["mean_r"], label=f"{label} mean", color=color)
        axes[2].plot(
            steps, result.history["min_r"], label=f"{label} minimum", color=color, ls="--"
        )
    for axis, title in zip(
        axes, ("Training objective", "Camera-split MSE", "Training-camera Pearson r")
    ):
        axis.axvline(cfg.STAGE1_STEPS, color="k", ls=":", lw=1, alpha=0.5)
        axis.set_xlabel("Step")
        axis.set_title(title)
        axis.legend()
        axis.grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- final evaluation --
    net.eval()
    # Final mesh = gate outcome: refined mesh under occlusion physics if Stage 2 was
    # accepted, else the exact Stage-1 mesh under the convex model (= the floor).
    pipeline.reset_occlusion(cfg.OCCLUSION_ENABLED and stage2_accepted)
    with torch.no_grad():
        if stage2_accepted:
            _, deformed_final = net(base_vertices, base_normals)
        else:
            deformed_final = verts_s1
        pred_norm = pipeline.predicted_lightcurves(deformed_final[0])
    print(f"Final mesh: {final_mesh_label} (occlusion physics: {bool(pipeline.state['active'])})")

    per_camera_df = compute_per_camera_metrics(observed_norm, pred_norm, camera_names)
    validation_names = {camera_names[index] for index in validation_indices.tolist()}
    per_camera_df["Split"] = [
        "validation" if name in validation_names else "train" for name in camera_names
    ]
    split_mse = {}
    for label, indices in (("train", train_indices), ("validation", validation_indices)):
        mse = compute_mse(observed_norm[indices], pred_norm[indices])
        split_mse[label] = float(mse)
        print(f"{label} MSE: {float(mse):.6f}")
    mean_r = float(per_camera_df["Pearson_R"].mean())
    min_r = float(per_camera_df["Pearson_R"].min())
    print(f"All-camera mean r={mean_r:.4f}, minimum r={min_r:.4f}")
    print(per_camera_df.sort_values("Pearson_R").to_string())
    per_camera_df.to_csv(results_dir / "per_camera_metrics.csv")

    target_cam = cfg.TARGET_CAM if cfg.TARGET_CAM in camera_names else camera_names[0]
    fig, _ = plot_lightcurve_comparison(
        obs_norm=observed_norm.detach().cpu().numpy(),
        pred_norm=pred_norm.detach().cpu().numpy(),
        camera_keys=camera_names,
        target_cam_id=target_cam,
    )
    fig.savefig(results_dir / "lightcurve_comparison.png", dpi=150)
    plt.close(fig)

    phase_deg = np.rad2deg(phases.detach().cpu().numpy())
    fig, _ = plot_camera_grid(observed_norm, pred_norm, camera_names, phase_axis=phase_deg)
    fig.savefig(results_dir / "camera_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------- mesh export + metrics --
    vertices_np = deformed_final[0].detach().cpu().numpy()
    recon_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    print(
        f"Final mesh ({final_mesh_label}) bounds: {recon_mesh.bounds}; "
        f"cylinder R={prior['R_CYLINDER']}, z in [-1, 1]"
    )

    reference_mesh = None
    if reference_stl_path_eval.exists():
        reference_mesh = normalize_mesh_to_challenge_cylinder(
            trimesh.load_mesh(reference_stl_path_eval),
            prior["R_CYLINDER"],
        )
    else:
        print(f"True STL unavailable at {reference_stl_path_eval}; skipping shape metrics.")

    recon_path = recon_dir / f"reconstructed_ast{asteroid_id}.stl"
    recon_mesh.export(recon_path)
    print(f"Saved final mesh ({final_mesh_label}): {recon_path}")

    stage1_to_refined = np.linalg.norm(vertices_np - verts_s1_np, axis=1)
    cylinder_scale = float(
        np.linalg.norm([prior["R_CYLINDER"], prior["R_CYLINDER"], 1.0])
    )
    print(
        f"Stage-1 -> final ({final_mesh_label}) vertex shift: "
        f"mean={stage1_to_refined.mean():.4f}, max={stage1_to_refined.max():.4f} "
        f"({100 * stage1_to_refined.mean() / cylinder_scale:.2f}% of the cylinder diagonal)"
    )

    shape_metrics: dict[str, dict[str, float]] = {}
    if reference_mesh is not None:
        stage1_mesh = trimesh.Trimesh(vertices=verts_s1_np, faces=faces_np, process=False)
        # Official challenge voxel measures (0 = identical, lower is better).
        print(
            f"Official voxel measures vs true STL (pitch={cfg.VOXEL_PITCH}); "
            "0 = identical, lower is better:"
        )
        for name, mesh in (
            ("convex_prior", reference_mesh_norm),
            ("stage1", stage1_mesh),
            (f"final:{final_mesh_label}", recon_mesh),
        ):
            measure1, measure2 = relative_volume_difference_voxelized(
                mesh, reference_mesh, pitch=cfg.VOXEL_PITCH
            )
            chamfer = mesh_reconstruction_metrics(
                mesh,
                reference_mesh,
                n_surface_samples=cfg.CHAMFER_SURFACE_SAMPLES,
                pitch=cfg.CHAMFER_PITCH,
            )["surface_chamfer"]
            shape_metrics[name] = {
                "measure1_one_minus_iou": measure1,
                "measure2_symmetric_difference": measure2,
                "surface_chamfer": chamfer,
            }
            print(
                f"  {name:14s} measure1 (1 - IoU)={measure1:.4f} | "
                f"measure2 (sym-diff/sum)={measure2:.4f} | surface Chamfer={chamfer:.4f}"
            )
        if cfg.REFERENCE_MESH_SOURCE == "true":
            print(
                "ORACLE RUN: true geometry entered training; these are not "
                "self-supervised reconstruction results."
            )
    else:
        print("True reference mesh unavailable: shape metrics were not computed.")

    metrics = {
        "asteroid_id": asteroid_id,
        "final_mesh_label": final_mesh_label,
        "stage2_accepted": stage2_accepted,
        "stage2_acceptance": cfg.STAGE2_ACCEPTANCE,
        "convex_floor": convex_floor,
        "stage2_fit": stage2_fit,
        "stage1_under_occlusion": stage1_under_occlusion,
        "stage1_best_step": stage1_result.best_step,
        "stage2_best_step": stage2_result.best_step,
        "final_lightcurve_mse": split_mse,
        "pearson_r": {"mean": mean_r, "min": min_r},
        "vertex_shift_stage1_to_final": {
            "mean": float(stage1_to_refined.mean()),
            "max": float(stage1_to_refined.max()),
            "percent_of_cylinder_diagonal": float(
                100 * stage1_to_refined.mean() / cylinder_scale
            ),
        },
        "shape_metrics_vs_true_stl": shape_metrics,
        "voxel_pitch": cfg.VOXEL_PITCH,
        "reference_mesh_source": cfg.REFERENCE_MESH_SOURCE,
        "reference_stl_path": str(reference_stl_path),
        "checkpoint_path": str(checkpoint_path),
        "reconstruction_path": str(recon_path),
        "training_config_stage1": asdict(config1),
        "training_config_stage2": asdict(config2),
        "model_config": {key: value for key, value in model_config.items()},
        "seed": cfg.SEED,
        "split_seed": cfg.SPLIT_SEED,
    }
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved figures: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
