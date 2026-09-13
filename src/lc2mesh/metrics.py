"""Lightcurve and mesh-reconstruction metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial import cKDTree


def mesh_reconstruction_metrics(
    predicted: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    n_surface_samples: int = 8192,
    pitch: float = 0.04,
    seed: int = 1729,
) -> dict[str, float]:
    """Evaluation-only surface distance and solid overlap in a shared world grid.

    Chamfer is the sum of two mean Euclidean nearest-sample distances, not
    squared distance or vertex-density-weighted distance. No rotation/scale
    fitting is done here. Voxel indices are computed from world-space centers,
    not each mesh's unrelated voxel-grid origin. Solid metrics require closed,
    consistently wound meshes; self-intersections can still invalidate filling.
    """
    if n_surface_samples < 1 or pitch <= 0:
        raise ValueError("sample count and voxel pitch must be positive")
    pred_points, _ = trimesh.sample.sample_surface(predicted, n_surface_samples, seed=seed)
    target_points, _ = trimesh.sample.sample_surface(target, n_surface_samples, seed=seed + 1)
    pred_distances = cKDTree(target_points).query(pred_points)[0]
    target_distances = cKDTree(pred_points).query(target_points)[0]
    result = {
        "surface_chamfer": float(pred_distances.mean() + target_distances.mean()),
        "surface_p95": float(max(np.quantile(pred_distances, 0.95), np.quantile(target_distances, 0.95))),
        "watertight": bool(predicted.is_watertight),
        "winding_consistent": bool(predicted.is_winding_consistent),
    }
    valid = all(mesh.is_watertight and mesh.is_winding_consistent for mesh in (predicted, target))
    if valid:
        def world_voxels(mesh):
            points = mesh.voxelized(pitch).fill().points
            return set(map(tuple, np.rint(points / pitch).astype(np.int64)))

        pred_voxels, target_voxels = world_voxels(predicted), world_voxels(target)
        intersection = len(pred_voxels & target_voxels)
        result["solid_dice"] = 2 * intersection / max(1, len(pred_voxels) + len(target_voxels))
        result["solid_iou"] = intersection / max(1, len(pred_voxels | target_voxels))
    else:
        result.update(solid_dice=float("nan"), solid_iou=float("nan"))
    return result


def compute_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all elements."""
    return torch.mean((y_pred - y_true) ** 2)


def compute_pearson_r(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Pearson correlation along ``dim`` (per-camera by default)."""
    yt = y_true
    yp = y_pred

    yt_center = yt - yt.mean(dim=dim, keepdim=True)
    yp_center = yp - yp.mean(dim=dim, keepdim=True)

    cov = torch.mean(yt_center * yp_center, dim=dim)
    yt_std = torch.sqrt(torch.mean(yt_center**2, dim=dim))
    yp_std = torch.sqrt(torch.mean(yp_center**2, dim=dim))
    return cov / (yt_std * yp_std + float(eps))


def _to_numpy_2d(x: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    """Convert tensor/array input to float32 NumPy matrix [C, T]."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [n_cameras, n_phases], got {arr.shape}")

    return arr.astype(np.float32, copy=False)


def compute_per_camera_metrics(
    obs_norm: np.ndarray | torch.Tensor,
    pred_norm: np.ndarray | torch.Tensor,
    camera_keys: Sequence[str],
) -> pd.DataFrame:
    """Compute per-camera MSE, MAE, and Pearson correlation."""
    obs = _to_numpy_2d(obs_norm, "obs_norm")
    pred = _to_numpy_2d(pred_norm, "pred_norm")

    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: obs {obs.shape} vs pred {pred.shape}")

    n_cams = obs.shape[0]
    if len(camera_keys) != n_cams:
        raise ValueError(
            f"camera_keys length mismatch: expected {n_cams}, got {len(camera_keys)}"
        )

    residual = pred - obs
    mse = np.mean(residual**2, axis=1)
    mae = np.mean(np.abs(residual), axis=1)

    obs_center = obs - obs.mean(axis=1, keepdims=True)
    pred_center = pred - pred.mean(axis=1, keepdims=True)
    cov = np.mean(obs_center * pred_center, axis=1)
    obs_std = np.sqrt(np.mean(obs_center**2, axis=1))
    pred_std = np.sqrt(np.mean(pred_center**2, axis=1))
    pearson_r = cov / np.maximum(obs_std * pred_std, 1e-12)

    return pd.DataFrame(
        {
            "MSE": mse,
            "MAE": mae,
            "Pearson_R": pearson_r,
        },
        index=pd.Index(camera_keys, name="Camera"),
    )
