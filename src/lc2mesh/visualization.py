"""Visualization utilities for lightcurve model diagnostics."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import torch

from lc2mesh.metrics import compute_per_camera_metrics


def _to_numpy_2d(x: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [n_cameras, n_phases], got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _to_numpy_mesh_vertices(x: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape [n_vertices, 3], got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _to_numpy_mesh_faces(x: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape [n_faces, 3], got {arr.shape}")
    return arr.astype(np.int64, copy=False)


def plot_interactive_mesh_3d(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    facet_areas: np.ndarray | torch.Tensor | None = None,
    *,
    title: str = "Interactive Asteroid Mesh",
    color_mode: str = "facet_area",
    colorscale: str = "Viridis",
    show_edges: bool = False,
):
    """Render an interactive 3D mesh using Plotly Mesh3d.

    Parameters
    ----------
    vertices:
        Vertex array with shape ``[n_vertices, 3]``.
    faces:
        Triangle index array with shape ``[n_faces, 3]``.
    facet_areas:
        Optional per-face values used for coloring. If omitted and
        ``color_mode='height'``, faces are colored by radial distance of their
        centers from mesh center.
    title:
        Figure title.
    color_mode:
        ``'facet_area'`` or ``'height'``.
    colorscale:
        Plotly colorscale name.
    show_edges:
        If True, overlays mesh edges.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure (call ``fig.show()`` in notebook).
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive 3D mesh visualization. "
            "Install it with `pip install plotly`."
        ) from exc

    v = _to_numpy_mesh_vertices(vertices, "vertices")
    f = _to_numpy_mesh_faces(faces, "faces")

    if v.shape[0] == 0 or f.shape[0] == 0:
        raise ValueError("vertices and faces must be non-empty")
    if int(f.min()) < 0 or int(f.max()) >= int(v.shape[0]):
        raise ValueError("faces contain out-of-range vertex indices")

    centers = v[f].mean(axis=1)

    intensity = None
    colorbar_title = None
    mode = str(color_mode).lower()
    if facet_areas is not None and mode == "facet_area":
        if isinstance(facet_areas, torch.Tensor):
            fa = facet_areas.detach().cpu().numpy()
        else:
            fa = np.asarray(facet_areas)
        fa = fa.reshape(-1).astype(np.float32, copy=False)
        if fa.shape[0] != f.shape[0]:
            raise ValueError(
                f"facet_areas must have length {f.shape[0]}, got {fa.shape[0]}"
            )
        intensity = fa
        colorbar_title = "Facet Area"
    elif mode == "height":
        mesh_center = v.mean(axis=0, keepdims=True)
        intensity = np.linalg.norm(centers - mesh_center, axis=1)
        colorbar_title = "Center Distance"

    mesh_kwargs = {
        "x": v[:, 0],
        "y": v[:, 1],
        "z": v[:, 2],
        "i": f[:, 0],
        "j": f[:, 1],
        "k": f[:, 2],
        "opacity": 1.0,
        "flatshading": False,
        "lighting": {
            "ambient": 0.35,
            "diffuse": 0.8,
            "specular": 0.35,
            "roughness": 0.7,
            "fresnel": 0.12,
        },
        "lightposition": {"x": 120.0, "y": 200.0, "z": 80.0},
        "hovertemplate": "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
    }

    if intensity is None:
        mesh_kwargs["color"] = "#c0b08a"
    else:
        mesh_kwargs["intensity"] = intensity
        mesh_kwargs["colorscale"] = colorscale
        mesh_kwargs["showscale"] = True
        mesh_kwargs["colorbar"] = {"title": colorbar_title}

    fig = go.Figure(data=[go.Mesh3d(**mesh_kwargs)])

    if show_edges:
        edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        xe: list[float | None] = []
        ye: list[float | None] = []
        ze: list[float | None] = []
        for a, b in edges:
            xe.extend([v[a, 0], v[b, 0], None])
            ye.extend([v[a, 1], v[b, 1], None])
            ze.extend([v[a, 2], v[b, 2], None])

        fig.add_trace(
            go.Scatter3d(
                x=xe,
                y=ye,
                z=ze,
                mode="lines",
                line={"color": "rgba(20,20,20,0.25)", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    bbox_min = v.min(axis=0)
    bbox_max = v.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    half_extent = 0.5 * float(np.max(bbox_max - bbox_min))
    half_extent = max(half_extent, 1e-6)

    x_range = [center[0] - half_extent, center[0] + half_extent]
    y_range = [center[1] - half_extent, center[1] + half_extent]
    z_range = [center[2] - half_extent, center[2] + half_extent]

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
        scene={
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 1},
            "camera": {
                "eye": {"x": 1.65, "y": 1.65, "z": 1.25},
                "projection": {"type": "perspective"},
            },
            "xaxis": {"title": "X", "range": x_range, "showspikes": False},
            "yaxis": {"title": "Y", "range": y_range, "showspikes": False},
            "zaxis": {"title": "Z", "range": z_range, "showspikes": False},
        },
    )
    return fig


def plot_interactive_mesh(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    *,
    title: str = "Interactive Asteroid Mesh",
    mesh_color: str = "#c0b08a",
    show_edges: bool = False,
):
    """Render a plain interactive 3D mesh (no facet-area colormap).

    This helper is intended for direct geometry inspection with mouse-based
    rotation/zoom/pan in notebooks.
    """
    fig = plot_interactive_mesh_3d(
        vertices=vertices,
        faces=faces,
        facet_areas=None,
        title=title,
        color_mode="facet_area",
        show_edges=show_edges,
    )
    fig.update_traces(
        selector={"type": "mesh3d"},
        color=mesh_color,
        intensity=None,
        showscale=False,
    )
    return fig


def _resolve_target_camera_id(
    camera_keys: Sequence[str],
    target_cam_id: int | str,
) -> int:
    if isinstance(target_cam_id, str):
        if target_cam_id not in camera_keys:
            raise ValueError(f"Unknown camera key '{target_cam_id}'")
        return int(list(camera_keys).index(target_cam_id))

    cam_id = int(target_cam_id)
    if cam_id < 0 or cam_id >= len(camera_keys):
        raise ValueError(
            f"target_cam_id must be in [0, {len(camera_keys) - 1}], got {cam_id}"
        )
    return cam_id


def plot_lightcurve_comparison(
    obs_norm: np.ndarray,
    pred_norm: np.ndarray,
    camera_keys: Sequence[str],
    target_cam_id: int | str = 0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot flattened curves with a non-overlapping zoom panel for one camera."""
    obs = np.asarray(obs_norm, dtype=np.float32)
    pred = np.asarray(pred_norm, dtype=np.float32)

    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: obs {obs.shape}, pred {pred.shape}")
    if obs.ndim != 2:
        raise ValueError(f"Expected [n_cams, n_phases], got {obs.shape}")

    n_cams, n_phases = obs.shape
    if len(camera_keys) != n_cams:
        raise ValueError(
            f"camera_keys length mismatch: expected {n_cams}, got {len(camera_keys)}"
        )

    cam_id = _resolve_target_camera_id(camera_keys, target_cam_id)
    flat_obs = obs.reshape(-1)
    flat_pred = pred.reshape(-1)
    x = np.arange(flat_obs.size, dtype=np.int64)

    fig = plt.figure(figsize=(20, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=(4.2, 1.8))
    ax = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])

    ax.plot(x, flat_obs, color="black", linewidth=1.0, label="Observed")
    ax.plot(x, flat_pred, color="#d95f02", linewidth=1.0, alpha=0.9, label="Forward Model")

    start = cam_id * n_phases
    stop = (cam_id + 1) * n_phases
    ax.axvspan(start, stop, color="#1f77b4", alpha=0.08)

    ax.set_title("Flattened Multi-Camera Lightcurve Comparison")
    ax.set_xlabel("Flattened sample index (all cameras concatenated)")
    ax.set_ylabel("Normalized flux")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    phase_idx = np.arange(n_phases)
    ax_zoom.plot(phase_idx, obs[cam_id], color="black", linewidth=1.0, label="Observed")
    ax_zoom.plot(phase_idx, pred[cam_id], color="#d95f02", linewidth=1.0, label="Forward Model")
    ax_zoom.set_title("Zoomed segment", fontsize=11)
    ax_zoom.set_xlabel("Phase sample")
    ax_zoom.set_ylabel("Normalized flux")
    ax_zoom.grid(alpha=0.25)

    low = float(min(obs[cam_id].min(), pred[cam_id].min()))
    high = float(max(obs[cam_id].max(), pred[cam_id].max()))
    pad = 0.06 * max(high - low, 1e-6)
    ax_zoom.set_xlim(0, n_phases - 1)
    ax_zoom.set_ylim(low - pad, high + pad)

    return fig, ax


def plot_camera_grid(
    obs_norm: np.ndarray,
    pred_norm: np.ndarray,
    camera_keys: Sequence[str],
    phase_axis: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot observed vs predicted lightcurves per camera in a dynamic grid."""
    obs = _to_numpy_2d(obs_norm, "obs_norm")
    pred = _to_numpy_2d(pred_norm, "pred_norm")

    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: obs {obs.shape} vs pred {pred.shape}")

    n_cams, n_phases = obs.shape
    if len(camera_keys) != n_cams:
        raise ValueError(
            f"camera_keys length mismatch: expected {n_cams}, got {len(camera_keys)}"
        )

    if phase_axis is None:
        x = np.arange(n_phases, dtype=np.float32)
        x_label = "Phase sample"
    else:
        x = np.asarray(phase_axis, dtype=np.float32).reshape(-1)
        if x.size != n_phases:
            raise ValueError(
                f"phase_axis length mismatch: expected {n_phases}, got {x.size}"
            )
        x_label = "Phase"

    metrics_df = compute_per_camera_metrics(obs, pred, camera_keys)

    n_cols = min(4, n_cams)
    n_rows = int(math.ceil(n_cams / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 2.8 * n_rows),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for i, cam in enumerate(camera_keys):
        ax = axes_flat[i]
        ax.plot(x, obs[i], color="black", linewidth=1.0, label="Observed")
        ax.plot(x, pred[i], color="#d95f02", linewidth=1.0, alpha=0.9, label="Predicted")
        ax.set_title(f"{cam} | r={metrics_df.loc[cam, 'Pearson_R']:.3f}", fontsize=9)
        ax.grid(alpha=0.25)

        if i % n_cols == 0:
            ax.set_ylabel("Normalized flux")
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel(x_label)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    for j in range(n_cams, axes_flat.size):
        axes_flat[j].axis("off")

    fig.suptitle("Per-Camera Lightcurve Fit Quality", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes_flat


# --- Rotating mesh video -----------------------------------------------------
# matplotlib depth-sorts every triangle per frame; a raw challenge STL is ~800k
# faces, so display meshes are decimated. Render-only: metrics never use these.
MAX_DISPLAY_FACES = 20000


def _display_mesh(mesh, max_display_faces: int = MAX_DISPLAY_FACES):
    """Decimate a high-poly mesh for fast matplotlib rendering (visual only)."""
    if len(mesh.faces) <= max_display_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=max_display_faces)
    except Exception as exc:  # no decimation backend -> fall back to the full mesh
        print(f"  (decimation unavailable: {exc}; rendering full {len(mesh.faces)} faces)")
        return mesh


def _shaded_facecolors(vertices, faces, base_hex, light_dir=(0.4, 0.3, 1.0), ambient=0.5, diffuse=0.5):
    """Fixed-light Lambert shading so the rotating mesh reads as a solid object."""
    from matplotlib import colors as mcolors

    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = np.asarray(light_dir, dtype=float)
    light /= np.linalg.norm(light)
    shade = ambient + diffuse * np.clip(normals @ light, 0.0, 1.0)
    base = np.asarray(mcolors.to_rgb(base_hex))
    return np.clip(shade[:, None] * base[None, :], 0.0, 1.0)


def _add_mesh(ax, mesh, base_hex, lim, background_color):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    collection = Poly3DCollection(
        vertices[faces],
        facecolors=_shaded_facecolors(vertices, faces, base_hex),
        edgecolors="none",
    )
    ax.add_collection3d(collection)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_facecolor(background_color)


def render_rotating_mesh_video(
    panels: Sequence[tuple[str, object, str]],
    video_path,
    *,
    title_prefix: str = "",
    limit: float = 1.0,
    n_frames: int = 60,
    fps: int = 10,
    elevation: float = 20.0,
    background_color: str = "white",
    max_display_faces: int = MAX_DISPLAY_FACES,
):
    """Write an mp4 of one or more meshes rotating a full turn around +z.

    ``panels`` is a sequence of ``(name, trimesh.Trimesh, hex_color)`` drawn
    side by side in a single figure.
    """
    import io

    import imageio.v2 as imageio

    panels = [(name, _display_mesh(mesh, max_display_faces), color) for name, mesh, color in panels]
    title_color = "white" if background_color == "black" else "black"

    fig = plt.figure(figsize=(6 * len(panels), 6), facecolor=background_color)
    axes = []
    for index, (name, mesh, color) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), index + 1, projection="3d")
        ax.set_title(f"{title_prefix}{name}", color=title_color)
        _add_mesh(ax, mesh, color, limit, background_color)
        axes.append(ax)

    # Force the ffmpeg backend and a broadly-compatible pixel format.
    writer = imageio.get_writer(
        str(video_path), format="FFMPEG", fps=fps, codec="libx264",
        macro_block_size=16, output_params=["-pix_fmt", "yuv420p"],
    )
    for frame_index in range(n_frames):
        azim = 360.0 * frame_index / n_frames
        for ax in axes:
            ax.view_init(elev=elevation, azim=azim)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor=fig.get_facecolor(), dpi=100)
        buffer.seek(0)
        writer.append_data(imageio.imread(buffer)[..., :3])
    writer.close()
    plt.close(fig)
    return video_path
