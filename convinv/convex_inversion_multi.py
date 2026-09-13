"""
Convex lightcurve inversion, extended to all four challenge_data lightcurve
kinds and to ground-truth evaluation.
"""

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize

pv.set_jupyter_backend("static")

from lc2mesh.constants import CAMERA_ANGLE_ALPHA
from lc2mesh.data import get_model_stl_path, load_data
from lc2mesh.eval import calculate_voxel_measure
from lc2mesh.mesh import normalize_mesh_to_challenge_cylinder

from convinv.convex_forward import get_unit_vectors
from convinv.minkowski import build_convex_mesh, minkowski_reconstruct, save_stl

# (label, azimuth_deg, elevation_deg) views used by save_reconstruction's
# multi-angle preview plot -- four around the equator plus a top-down view.
DEFAULT_RECON_VIEWS = [
    ("front", 0, 20),
    ("right", 90, 20),
    ("back", 180, 20),
    ("left", 270, 20),
    ("top", 0, 90),
]

# drop every "_hor2" column
CAMERA_ANGLE_ALPHA_DEDUPED: dict[str, float] = {
    cam: alpha for cam, alpha in CAMERA_ANGLE_ALPHA.items() if not cam.endswith("_hor2")
}


def _cam_group(cam: str) -> str:
    """
    Groups cameras by azimuthal angle for per-angle scattering-parameter
    fitting
    """
    if cam in ("0_hor1", "0_hor2"):
        return cam
    return cam.split("_")[0]


def _facet_normals_and_areas(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Per-facet outward unit normals and areas."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    cross = np.cross(e1, e2)
    mag = np.linalg.norm(cross, axis=1)
    normals = cross / mag[:, None]
    areas = 0.5 * mag
    return normals, areas


def _calculate_voxel_measure_solid(
    true_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, pitch: float = 0.02
) -> float:
    """
    Solid-volume analogue of `lc2mesh.eval.calculate_voxel_measure`. That
    function's `mesh.voxelized(pitch)`

    Same symmetric-difference / (|A|+|B|) Dice-style formula as
    `calculate_voxel_measure`, but on `.voxelized(pitch).fill()` (solid)
    voxel sets instead
    """
    true_voxels = set(map(tuple, true_mesh.voxelized(pitch).fill().sparse_indices))
    pred_voxels = set(map(tuple, pred_mesh.voxelized(pitch).fill().sparse_indices))
    if len(true_voxels) == 0 and len(pred_voxels) == 0:
        return 0.0
    mismatch_count = len(true_voxels.symmetric_difference(pred_voxels))
    total = len(true_voxels) + len(pred_voxels)
    return float(np.clip(1.0 - mismatch_count / total, 0.0, 1.0))


def _rrmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Root-relative-mean-square-error: RMSE(pred, obs) / RMS(obs)."""
    pred, obs = np.asarray(pred, dtype=float), np.asarray(obs, dtype=float)
    denom = np.sqrt(np.mean(obs**2))
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.sqrt(np.mean((pred - obs) ** 2)) / denom)


def _corr(pred: np.ndarray, obs: np.ndarray) -> float:
    pred, obs = np.asarray(pred, dtype=float), np.asarray(obs, dtype=float)
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(obs))):
        return float("nan")
    if np.std(pred) == 0 or np.std(obs) == 0:
        return float("nan")
    return float(np.corrcoef(pred, obs)[0, 1])


def _circular_align_all(
    curves_a: dict[str, np.ndarray], curves_b: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], int]:
    """
    Find the single integer circular shift (applied to every curve in
    `curves_a`) that maximises total cross-correlation with the matching
    curve in `curves_b`, summed over all shared cameras

    Brute-force over all integer lags -- n_phases is at most a few hundred
    to ~1000 samples, so this is cheap.
    """
    common = [c for c in curves_a if c in curves_b]
    n = len(next(iter(curves_a.values())))
    b0 = {c: curves_b[c] - np.nanmean(curves_b[c]) for c in common}

    best_shift, best_score = 0, -np.inf
    for k in range(n):
        score = 0.0
        for c in common:
            a0 = np.roll(curves_a[c], k)
            a0 = a0 - np.nanmean(a0)
            if not (np.all(np.isfinite(a0)) and np.all(np.isfinite(b0[c]))):
                continue
            score += float(np.dot(a0, b0[c]))
        if score > best_score:
            best_score, best_shift = score, k

    shifted = {c: np.roll(curves_a[c], best_shift) for c in curves_a}
    return shifted, best_shift


class ConvexInversionMulti:
    def __init__(self, cylinder_radius=1.05, max_faces=1500, initial_shape="cylinder"):
        """
        `initial_shape`:
        - "cylinder": an icosphere passed through `normalize_mesh_to_challenge_cylinder`,
          which forces z to span exactly [-1, 1] while scaling xy to
          `cylinder_radius`.
        - "sphere": a true isotropic ball, radius `cylinder_radius` in x, y,
          AND z (z/xy ratio 1.0) -- unlike "cylinder", not pre-shaped by the
          asteroid's own bounding-cylinder aspect ratio.
        """
        self.cylinder_radius = cylinder_radius

        subdiv = np.floor(np.log(max_faces / 20) / np.log(4)).astype(int)
        if subdiv < 0:
            subdiv = 0
        sphere_mesh = trimesh.creation.icosphere(subdivisions=subdiv)
        print(f"Using {len(sphere_mesh.faces)} faces")

        if initial_shape == "cylinder":
            init_mesh = normalize_mesh_to_challenge_cylinder(
                sphere_mesh, target_R=cylinder_radius
            )
        elif initial_shape == "sphere":
            init_mesh = sphere_mesh.copy()
            init_mesh.apply_scale(cylinder_radius)
        else:
            raise ValueError(
                f"initial_shape must be 'cylinder' or 'sphere', got {initial_shape!r}"
            )

        self.normals, self.areas_init = _facet_normals_and_areas(init_mesh)
        self.n_facets = len(self.areas_init)
        self.areas_init_total = np.sum(self.areas_init)

    def load_lightcurve_data(
        self, model_num, origin="simulated", curve_type="intensity"
    ):
        """
        Load lightcurve data for any of the four `origin x curve_type`
        combinations (see module docstring for the camera-set handling
        difference between "simulated" and "real").

        In self:
        - self.obs_norm: normalized lightcurve data (n_cams, n_phases)
        - self.e_sun: sun direction vector
        - self.CAM_DICT / self.cam_keys / self.n_cams: camera geometry used
        - self.phases: rotational phases, one per data frame (n_phases,)
        - self.origin, self.curve_type, self.asteroid
        """
        lightcurve_ds = load_data(model_num, origin, curve_type)

        if origin == "simulated":
            self.CAM_DICT = CAMERA_ANGLE_ALPHA_DEDUPED
        elif origin == "real":
            self.CAM_DICT = CAMERA_ANGLE_ALPHA
        else:
            raise ValueError(f"origin must be 'simulated' or 'real', got {origin!r}")
        self.cam_keys = list(self.CAM_DICT.keys())
        self.n_cams = len(self.cam_keys)

        # One rotational phase per data frame -- Real data has a
        # different frame count (e.g. 842 for asteroid 2) than blender's
        # fixed 360, so n_phases is taken from the data rather than hardcoded.
        self.n_phases = len(lightcurve_ds.data)
        self.phases = np.linspace(0, 2 * np.pi, self.n_phases, endpoint=False)

        self.obs_norm = np.empty((self.n_cams, self.n_phases))
        for i, cam in enumerate(self.cam_keys):
            lc = lightcurve_ds.data[cam].values.astype(np.float64)
            self.obs_norm[i] = (
                lc  # already per-camera mean-normalized in challenge_data
            )

        print(
            f"Observations ({origin}/{curve_type}): {self.n_cams} cameras x "
            f"{self.n_phases} phases = {self.n_cams * self.n_phases} data points"
        )
        self.e_sun = np.array([-1.0, 0.0, 0.0])
        self.origin = origin
        self.curve_type = curve_type
        self.asteroid = model_num
        # scattering params from a previous fit no longer apply to new data
        self.scattering_params = None
        return

    def load_lightcurve_data_joint(self, model_num, origin="simulated"):
        """
        Load BOTH curve types (intensity + binary) for the same asteroid/
        origin, for the "extended" forward operator that fits one shared
        set of facet areas against both simultaneously (`fit_areas_joint`)
        instead of one curve type at a time. Mixing origins (e.g. simulated
        intensity + real binary) isn't supported here

        Populates self.channels: a list of two dicts (curve_type
        "intensity" then "binary"), each holding that channel's own
        obs_norm/phases/n_phases
        """
        self.channels = []
        cam_dict = None
        for curve_type in ("intensity", "binary"):
            lightcurve_ds = load_data(model_num, origin, curve_type)

            if origin == "simulated":
                cam_dict = CAMERA_ANGLE_ALPHA_DEDUPED
            elif origin == "real":
                cam_dict = CAMERA_ANGLE_ALPHA
            else:
                raise ValueError(
                    f"origin must be 'simulated' or 'real', got {origin!r}"
                )
            cam_keys = list(cam_dict.keys())
            n_cams = len(cam_keys)

            n_phases = len(lightcurve_ds.data)
            phases = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)

            obs_norm = np.empty((n_cams, n_phases))
            for i, cam in enumerate(cam_keys):
                obs_norm[i] = lightcurve_ds.data[cam].values.astype(np.float64)

            print(
                f"Observations ({origin}/{curve_type}): {n_cams} cameras x "
                f"{n_phases} phases = {n_cams * n_phases} data points"
            )

            self.channels.append(
                {
                    "curve_type": curve_type,
                    "obs_norm": obs_norm,
                    "phases": phases,
                    "n_phases": n_phases,
                }
            )

        self.CAM_DICT = cam_dict
        self.cam_keys = list(cam_dict.keys())
        self.n_cams = len(self.cam_keys)
        self.e_sun = np.array([-1.0, 0.0, 0.0])
        self.origin = origin
        self.curve_type = "joint"
        self.asteroid = model_num
        self.scattering_params = None
        return

    def _rotated_mu_mu0(self, cam, cos_p=None, sin_p=None, normals=None):
        """mu/mu0 for `cam` against `normals` (default self.normals -- the
        reconstruction's own facet normals; pass an arbitrary shape's
        normals, e.g. a ground-truth mesh's, to evaluate against that
        instead)."""
        if cos_p is None:
            cos_p, sin_p = np.cos(self.phases), np.sin(self.phases)
        if normals is None:
            normals = self.normals
        phi_deg = float(cam.split("_")[0])
        alpha_deg = self.CAM_DICT[cam]
        _, e_obs = get_unit_vectors(phi_deg, alpha_deg)
        e_obs = np.asarray(e_obs, dtype=np.float64)

        n = len(cos_p)
        e_sun_rot = np.column_stack(
            [
                cos_p * self.e_sun[0] - sin_p * self.e_sun[1],
                sin_p * self.e_sun[0] + cos_p * self.e_sun[1],
                np.full(n, self.e_sun[2]),
            ]
        )
        e_obs_rot = np.column_stack(
            [
                cos_p * e_obs[0] - sin_p * e_obs[1],
                sin_p * e_obs[0] + cos_p * e_obs[1],
                np.full(n, e_obs[2]),
            ]
        )
        mu0 = e_sun_rot @ normals.T
        mu = e_obs_rot @ normals.T
        return mu0, mu

    def _scattering_S(self, mu, mu0, value, curve_type=None):
        """
        curve_type == "intensity": LSL law (eq. 2 of arXiv:2502.16455),
        `value` = lambert_weight.

        curve_type == "binary": mu0-threshold illumination law,
        `value` = mu0_threshold. A facet's projected area (area * mu)
        is counted iff it's visible (mu > 0) and its sun angle clears the threshold.

        `curve_type` defaults to self.curve_type; pass it explicitly for
        joint-mode calls (self.curve_type is "joint" there, so the
        per-channel scattering law must come from the caller -- see
        `build_forward_joint`/`fit_scattering_params_joint`).
        """
        curve_type = curve_type if curve_type is not None else self.curve_type
        if curve_type == "intensity":
            mask = (mu0 > 0) & (mu > 0)
            return np.where(
                mask, mu * mu0 / np.maximum(mu + mu0, 1e-30) + value * mu * mu0, 0.0
            )
        elif curve_type == "binary":
            mask = (mu > 0) & (mu0 > value)
            return np.where(mask, mu, 0.0)
        else:
            raise ValueError(f"Unknown curve_type {curve_type!r}")

    def _simulate_curves_for_shape(
        self,
        normals,
        areas,
        params=None,
        default=0.1,
        phase_batch_size=64,
        group_fn=_cam_group,
        curve_type=None,
        phases=None,
        n_phases=None,
    ):
        """
        Forward-simulate normalized lightcurves for an arbitrary shape
        (`normals`, `areas`) using self.cam_keys/self.e_sun and the
        scattering law implied by `curve_type` (`_scattering_S`),
        processing phases in batches of `phase_batch_size` so peak memory
        is O(phase_batch_size * n_facets), independent of n_phases or facet
        count.

        Needed for ground-truth STLs, which are full-resolution meshes
        (e.g. asteroid 2 has ~538k facets, asteroid 1 ~800k) -- computing a
        dense (n_phases, n_facets) array per camera in one shot for those
        (especially against real data's 842-frame phase grid) can exceed
        available memory.

        `curve_type`/`phases`/`n_phases` default to self.curve_type/
        self.phases/self.n_phases (original single-channel behavior); pass
        them explicitly (e.g. from one of self.channels) for a joint-mode
        channel -- see `evaluate_against_ground_truth_joint`.
        """
        curve_type = curve_type if curve_type is not None else self.curve_type
        phases = self.phases if phases is None else phases
        n_phases = self.n_phases if n_phases is None else n_phases
        params = params or {}
        cos_p_full, sin_p_full = np.cos(phases), np.sin(phases)
        curves = {}
        for cam in self.cam_keys:
            value = params.get(group_fn(cam), default)

            raw = np.empty(n_phases)
            for start in range(0, n_phases, phase_batch_size):
                end = min(start + phase_batch_size, n_phases)
                cos_p, sin_p = cos_p_full[start:end], sin_p_full[start:end]
                mu0, mu = self._rotated_mu_mu0(cam, cos_p, sin_p, normals=normals)
                S = self._scattering_S(mu, mu0, value, curve_type=curve_type)
                raw[start:end] = (S * areas).sum(axis=1)

            m = raw.mean()
            curves[cam] = raw / m if m > 0 else np.full(n_phases, np.nan)
        return curves

    def fit_scattering_params(
        self,
        grid=None,
        normals=None,
        areas=None,
        verbose=True,
        group_fn=_cam_group,
        curve_type=None,
        phases=None,
        obs_norm=None,
        n_phases=None,
    ):
        """
        Grid-search the forward model's single free scattering parameter
        (lambert_weight for intensity, mu0_threshold for binary) separately
        per camera group (`group_fn`, default `_cam_group` = azimuth-based
        grouping), following the same methodology as
        `fit_and_simulate_by_group`/`fit_param` in convex_forward_sim_stl.py.
        Call after `load_lightcurve_data` and before `build_forward`, and
        pass the result (or nothing -- it's cached on self) into
        `build_forward(params=..., group_fn=...)` -- `group_fn` must match
        between the two calls, or `build_forward`'s lookups against the
        fitted `params` dict's keys will silently miss and fall back to
        `default`.

        By default, scores candidate values using the current initial-guess
        shape (self.normals / self.areas_init) as a representative facet
        distribution -- a proxy, since the true shape is unknown at this
        stage. Pass `normals`/`areas` (e.g. a known ground-truth mesh's, via
        `_facet_normals_and_areas`) to fit against that instead -- useful
        when validating against an asteroid with a known shape

        `group_fn`: maps a camera name to its group name. `_cam_group`
        (default) groups by azimuth, keeping `0_hor1`/`0_hor2` individual
        (see its docstring). Pass e.g. `lambda cam: cam` to fit every
        camera fully independently -- useful when a shared group's fit
        looks like a poor compromise for some of its cameras.

        `curve_type`/`phases`/`obs_norm`/`n_phases`: override self.curve_type
        /self.phases/self.obs_norm/self.n_phases -- all four default to None
        (use self.*, original single-channel behavior, and cache the result
        as self.scattering_params). Pass them explicitly (e.g. from one of
        self.channels) to fit a single channel's parameter in joint mode
        without touching self or self.scattering_params -- see
        `fit_scattering_params_joint`.
        """
        store = (
            curve_type is None
            and phases is None
            and obs_norm is None
            and n_phases is None
        )
        if not hasattr(self, "obs_norm") and store:
            raise ValueError(
                "Lightcurve data not loaded. Call load_lightcurve_data() first."
            )
        if curve_type is None:
            curve_type = self.curve_type
        if phases is None:
            phases = self.phases
        if obs_norm is None:
            obs_norm = self.obs_norm
        if n_phases is None:
            n_phases = self.n_phases
        if normals is None:
            normals = self.normals
        if areas is None:
            areas = self.areas_init
        if grid is None:
            grid = (
                np.linspace(0.0, 30.0, 150)
                if curve_type == "intensity"
                else np.linspace(0.02, 0.98, 49)
            )
        param_name = "lambert_weight" if curve_type == "intensity" else "mu0_threshold"

        cos_p, sin_p = np.cos(phases), np.sin(phases)
        groups: dict[str, list[int]] = {}
        for i, cam in enumerate(self.cam_keys):
            groups.setdefault(group_fn(cam), []).append(i)

        params: dict[str, float] = {}
        for group_name, idxs in sorted(
            groups.items(), key=lambda kv: (float(kv[0].split("_")[0]), kv[0])
        ):
            best_val, best_mse = float(grid[0]), np.inf
            for val in grid:
                mses = []
                for i in idxs:
                    cam = self.cam_keys[i]
                    mu0, mu = self._rotated_mu_mu0(cam, cos_p, sin_p, normals=normals)
                    S = self._scattering_S(mu, mu0, val, curve_type=curve_type)
                    raw = (S * areas).sum(axis=1)
                    m = raw.mean()
                    if m <= 0:
                        mses = None
                        break
                    mses.append(np.mean((raw / m - obs_norm[i]) ** 2))
                if mses is None:
                    continue
                mse = float(np.mean(mses))
                if mse < best_mse:
                    best_mse, best_val = mse, float(val)
            params[group_name] = best_val
            if verbose:
                label = f"{group_name} deg" if group_name.isdigit() else group_name
                print(
                    f"  {label:>9s} ({len(idxs):2d} cams): best {param_name} = {best_val:.3f}  (mse={best_mse:.6f})"
                )

        if store:
            self.scattering_params = params
        return params

    def fit_scattering_params_joint(
        self,
        grid_intensity=None,
        grid_binary=None,
        normals=None,
        areas=None,
        verbose=True,
        group_fn=_cam_group,
    ):
        """
        Joint-mode analogue of `fit_scattering_params`: fits each channel's
        scattering-law parameter separately.

        Stores the result on each entry of self.channels as
        channel["scattering_params"], and also self.scattering_params as
        {"intensity": {...}, "binary": {...}} for inspection/reuse (e.g.
        hardcoding into a cheap follow-up script, as done elsewhere in this
        investigation to avoid re-fitting).
        """
        if not hasattr(self, "channels"):
            raise ValueError(
                "Joint lightcurve data not loaded. Call load_lightcurve_data_joint() first."
            )
        grids = {"intensity": grid_intensity, "binary": grid_binary}
        all_params = {}
        for channel in self.channels:
            ct = channel["curve_type"]
            if verbose:
                print(f"\nFitting {ct} scattering parameter:")
            params = self.fit_scattering_params(
                grid=grids[ct],
                normals=normals,
                areas=areas,
                verbose=verbose,
                group_fn=group_fn,
                curve_type=ct,
                phases=channel["phases"],
                obs_norm=channel["obs_norm"],
                n_phases=channel["n_phases"],
            )
            channel["scattering_params"] = params
            all_params[ct] = params
        self.scattering_params = all_params
        return all_params

    def set_scattering_params(self, value, group_fn=_cam_group, curve_type=None):
        """
        Set the same scattering-law parameter value (lambert_weight for
        intensity, mu0_threshold for binary) for every camera, instead of
        grid-searching it per camera group with `fit_scattering_params` --
        e.g. `set_scattering_params(0.2)` to fix lambert_weight=0.2
        everywhere.

        Populates self.scattering_params as a {group_name: value} dict, all
        groups mapping to the same `value`, using `group_fn` (must match
        whatever `build_forward` is later called with) to enumerate group
        names from self.cam_keys -- same shape `fit_scattering_params`
        produces, so it's a drop-in replacement feeding straight into
        `build_forward(params=...)` or the cached `self.scattering_params`.

        In joint mode (self.channels populated via
        `load_lightcurve_data_joint`), sets both channels' `scattering_params`
        and self.scattering_params becomes {"intensity": {...}, "binary":
        {...}}, matching `fit_scattering_params_joint`'s output shape. Pass
        `curve_type` ("intensity" or "binary") to set only that one channel,
        leaving the other channel's existing params (if any) untouched.
        """
        if not hasattr(self, "cam_keys"):
            raise ValueError(
            "Lightcurve data not loaded. Call load_lightcurve_data() "
            "(or load_lightcurve_data_joint()) first."
            )
        groups = sorted({group_fn(cam) for cam in self.cam_keys})
        params = {group: float(value) for group in groups}

        if hasattr(self, "channels"):
            target_types = (
            [curve_type] if curve_type else [c["curve_type"] for c in self.channels]
            )
            all_params = dict(getattr(self, "scattering_params", None) or {})
            for channel in self.channels:
                if channel["curve_type"] in target_types:
                    channel["scattering_params"] = params
                    all_params[channel["curve_type"]] = params
            self.scattering_params = all_params
            print(
                f"Set scattering param = {value} for {len(groups)} camera group(s) "
                f"({', '.join(target_types)})"
            )
        else:
            self.scattering_params = params
            print(f"Set scattering param = {value} for {len(groups)} camera group(s)")
        return params


    def build_forward(
        self,
        params=None,
        default=0.1,
        group_fn=_cam_group,
        curve_type=None,
        phases=None,
        n_phases=None,
    ):
        """
        Build the forward matrix A, using the LSL law for intensity data or
        the mu0-threshold illumination law for binary data (see
        `_scattering_S`), with a scattering-law parameter value per camera
        group.

        `params`: {group_name: value} dict (see `fit_scattering_params`,
        `_cam_group`). Defaults to `self.scattering_params` if that was
        already fit, else falls back to `default` for every camera.

        `group_fn`: must match whatever grouping `params` was fit with
        (see `fit_scattering_params`'s `group_fn`) -- e.g. both `None`/
        default (`_cam_group`), or both `lambda cam: cam`.

        `curve_type`/`phases`/`n_phases`: override self.curve_type/
        self.phases/self.n_phases.
        """
        store = curve_type is None and phases is None and n_phases is None
        if not hasattr(self, "obs_norm") and not hasattr(self, "channels"):
            raise ValueError("Lightcurve data not loaded.")
        if curve_type is None:
            curve_type = self.curve_type
        if phases is None:
            phases = self.phases
        if n_phases is None:
            n_phases = self.n_phases
        if params is None and store:
            params = getattr(self, "scattering_params", None)

        cos_p, sin_p = np.cos(phases), np.sin(phases)
        A_blocks = []
        used_params = {}
        for cam in self.cam_keys:
            value = params.get(group_fn(cam), default) if params else default
            used_params[cam] = value
            mu0, mu = self._rotated_mu_mu0(cam, cos_p, sin_p)
            A_blocks.append(self._scattering_S(mu, mu0, value, curve_type=curve_type))

        A = np.vstack(A_blocks)  # (n_cams * n_phases, n_facets)
        A_3d = A.reshape(self.n_cams, n_phases, self.n_facets)
        A_mean = A_3d.mean(axis=1)  # (n_cams, n_facets)

        if store:
            self.A, self.A_3d, self.A_mean, self.scattering_params_used = (
                A,
                A_3d,
                A_mean,
                used_params,
            )
            print(
                f"Forward matrix A ({self.origin}/{curve_type}): {self.A.shape}  ({self.A.nbytes / 1e6:.1f} MB)"
            )
            return
        print(
            f"Forward matrix A ({self.origin}/{curve_type}): {A.shape}  ({A.nbytes / 1e6:.1f} MB)"
        )
        return {
            "A": A,
            "A_3d": A_3d,
            "A_mean": A_mean,
            "scattering_params_used": used_params,
        }

    def build_forward_joint(self, params=None, default=0.1, group_fn=_cam_group):
        """
        Joint-mode analogue of `build_forward`: builds each channel's own
        forward matrix (different scattering law and phase grid per curve
        type, see `build_forward`), stored on that channel dict (keys "A",
        "A_3d", "A_mean", "scattering_params_used").

        `params`: optional {"intensity": {...}, "binary": {...}} dict
        overriding each channel's already-fitted
        self.channels[i]["scattering_params"] (from
        `fit_scattering_params_joint`) -- e.g. to reuse previously-fitted
        values without re-fitting.
        """
        if not hasattr(self, "channels"):
            raise ValueError(
                "Joint lightcurve data not loaded. Call load_lightcurve_data_joint() first."
            )
        params = params or {}
        total_bytes = 0
        for channel in self.channels:
            ct = channel["curve_type"]
            channel_params = (
                params.get(ct)
                if params.get(ct) is not None
                else channel.get("scattering_params")
            )
            built = self.build_forward(
                params=channel_params,
                default=default,
                group_fn=group_fn,
                curve_type=ct,
                phases=channel["phases"],
                n_phases=channel["n_phases"],
            )
            channel.update(built)
            total_bytes += built["A"].nbytes
        print(f"Joint forward matrices built: {total_bytes / 1e6:.1f} MB total")
        return

    def cost_and_grad(
        self,
        a,
        lambda_s,
        _use_gpu,
        lambda_a=None,
        parametrization="log",
        cam_weights=None,
        l1_regularization=0.0,
    ):
        """
        `lambda_s`: either a scalar (uniform regularization, the original
        behavior) or a length-3 (lambda_x, lambda_y, lambda_z) sequence for
        per-axis convexity regularization -- see `fit_areas`.

        `lambda_a`:  scalar for area regularization. If None, lambda_a = max(lambda_s). 
        Controls that the total area of the shape is not too far from the initial guess,
        which is need to avoid convexity regularization from shrinking the shape.

        `l1_regularization`: extra `l1_regularization * sum(area)` penalty
        added to the cost (0.0 = off, original behavior).

        `parametrization`: "log" (default) treats `a` as log(area) (areas
        g = exp(a)), enforcing positivity implicitly. "direct" instead treats `a`
        as the area itself (g = a), for use with an explicit bound
        constraint (see `fit_areas`) rather than an implicit exp() one,

        `cam_weights`: optional (n_cams,) array of per-camera weights for a
        weighted least-squares data-fit term (cost = 0.5 * sum_c w_c *
        sum_p res[c,p]^2, `self._xp`-agnostic). None = uniform weight 1
        (original behavior).
        """
        a_xp = self._xp.asarray(a, dtype=self._xp.float64)
        if parametrization == "log":
            g = self._xp.exp(a_xp)
        elif parametrization == "direct":
            g = a_xp
        else:
            raise ValueError(
                f"parametrization must be 'log' or 'direct', got {parametrization!r}"
            )

        Ag = self._A @ g
        Ag3d = Ag.reshape(self.n_cams, self.n_phases)
        m = Ag3d.mean(axis=1)

        norm_model = Ag3d / m[:, None]
        res = self._obs_norm - norm_model

        if cam_weights is None:
            wres = res
        else:
            w = self._xp.asarray(cam_weights, dtype=self._xp.float64)
            wres = w[:, None] * res

        cost = float(0.5 * self._xp.dot(wres.ravel(), res.ravel()))
        if _use_gpu and hasattr(cost, "get"):
            cost = cost.get()

        term1 = self._A.T @ (wres / m[:, None]).ravel()
        if cam_weights is None:
            term2 = self._A_mean.T @ (self._xp.sum(norm_model * res, axis=1) / m)
        else:
            term2 = self._A_mean.T @ (w * self._xp.sum(norm_model * res, axis=1) / m)
        grad_g = -(
            term1 - term2
        )  # gradient w.r.t. area (g), before any da/dg chain rule

        # conv_vec is the closure defect (x, y, z) = sum_i area_i * normal_i,
        # which should vanish for a closed convex polyhedron. lambda_s
        # broadcasts a scalar to (3,) (original uniform penalty
        # lambda_s * ||conv_vec||^2) or is already a per-axis (3,) weight --
        # either way cost = conv_vec . (lambda_vec * conv_vec).
        conv_vec = self._normals.T @ g
        lambda_vec = self._xp.broadcast_to(
            self._xp.asarray(lambda_s, dtype=self._xp.float64), (3,)
        )
        weighted_conv = lambda_vec * conv_vec
        cost += float(self._xp.dot(conv_vec, weighted_conv))
        grad_g += 2.0 * (self._normals @ weighted_conv)

        if lambda_a is None:
            lambda_a = float(self._xp.max(lambda_vec))
        g_total = self._xp.sum(g) 
        cost += float(lambda_a * (g_total - self.areas_init_total)**2)
        grad_g += 2.0 * lambda_a * (g_total-self.areas_init_total)

        if l1_regularization:
            cost += float(l1_regularization * self._xp.sum(g))
            grad_g += l1_regularization

        # chain rule: d(cost)/da = d(cost)/dg * dg/da. dg/da = g for
        # parametrization="log" (g=exp(a)); dg/da = 1 for "direct" (g=a).
        grad = g * grad_g if parametrization == "log" else grad_g

        if _use_gpu:
            return float(cost), grad.get().astype(np.float64)
        return float(cost), np.asarray(grad, dtype=np.float64)

    def cost_and_grad_joint(
        self,
        a,
        lambda_s,
        _use_gpu,
        lambda_a=None,
        parametrization="log",
        cam_weights=None,
        channel_weights=None,
    ):
        """
        Joint-mode analogue of `cost_and_grad`: sums the data-fit cost/
        gradient across all channels in self.channels (intensity + binary,
        see `load_lightcurve_data_joint`/`build_forward_joint`), each with
        its own forward matrix/phase grid/scattering law but the same
        shared area vector `g`.

        `channel_weights`: optional {"intensity": w, "binary": w} dict to
        balance the two channels' contribution to the total cost -- useful
        since they have different phase-grid sizes (so raw summed-residual
        magnitude differs) and different scattering-law units. None
        (default) is weight 1.0 for both.
        """
        a_xp = self._xp.asarray(a, dtype=self._xp.float64)
        if parametrization == "log":
            g = self._xp.exp(a_xp)
        elif parametrization == "direct":
            g = a_xp
        else:
            raise ValueError(
                f"parametrization must be 'log' or 'direct', got {parametrization!r}"
            )

        channel_weights = channel_weights or {}
        cost = 0.0
        grad_g = self._xp.zeros_like(g)

        for channel in self.channels:
            ct = channel["curve_type"]
            cw = channel_weights.get(ct, 1.0)
            A, A_mean, obs_norm = (
                channel["_A"],
                channel["_A_mean"],
                channel["_obs_norm"],
            )
            n_phases = channel["n_phases"]

            Ag = A @ g
            Ag3d = Ag.reshape(self.n_cams, n_phases)
            m = Ag3d.mean(axis=1)

            norm_model = Ag3d / m[:, None]
            res = obs_norm - norm_model

            if cam_weights is None:
                wres = res
            else:
                w = self._xp.asarray(cam_weights, dtype=self._xp.float64)
                wres = w[:, None] * res

            cost += cw * float(0.5 * self._xp.dot(wres.ravel(), res.ravel()))

            term1 = A.T @ (wres / m[:, None]).ravel()
            if cam_weights is None:
                term2 = A_mean.T @ (self._xp.sum(norm_model * res, axis=1) / m)
            else:
                term2 = A_mean.T @ (w * self._xp.sum(norm_model * res, axis=1) / m)
            grad_g += cw * -(term1 - term2)

        conv_vec = self._normals.T @ g
        lambda_vec = self._xp.broadcast_to(
            self._xp.asarray(lambda_s, dtype=self._xp.float64), (3,)
        )
        weighted_conv = lambda_vec * conv_vec
        cost += float(self._xp.dot(conv_vec, weighted_conv))
        grad_g += 2.0 * (self._normals @ weighted_conv)

        if lambda_a is None:
            lambda_a = float(self._xp.max(lambda_vec))
        g_total = self._xp.sum(g) 
        cost += float(lambda_a * (g_total - self.areas_init_total)**2)
        grad_g += 2.0 * lambda_a * (g_total-self.areas_init_total)

        grad = g * grad_g if parametrization == "log" else grad_g

        if _use_gpu:
            return float(cost), grad.get().astype(np.float64)
        return float(cost), np.asarray(grad, dtype=np.float64)

    def fit_areas(
        self,
        convexity_regularization=0.1,
        scale_penalty=None,
        optimizer="cg",
        max_iter=10000,
        gtol=1e-8,
        avoid_gpu=False,
        parametrization="log",
        direct_area_floor=None,
        cam_weights=None,
        l1_regularization=0.0,
    ):
        """
        `l1_regularization`: see `cost_and_grad` -- an extra
        `l1_regularization * sum(area)` sparsity-inducing penalty, 0.0
        (default) = off, original behavior.

        `convexity_regularization`: either a scalar (applied uniformly to
        the closure-defect vector's x/y/z components, the original
        behavior) or a (lambda_x, lambda_y, lambda_z) sequence to weight
        them separately.

        `parametrization`: "log" (default, original behavior) optimizes
        a=log(area). "direct" optimizes area itself, positivity enforced
        via an explicit lower bound (`direct_area_floor`, default
        1e-6 * mean(areas_init)) instead of exp() -- see `cost_and_grad`
        for why this changes the optimizer's step dynamics. Bounds require
        a bounded method, so `optimizer` is forced to "lbfgs" (L-BFGS-B)
        for this mode regardless of what's passed.

        `cam_weights`: optional (n_cams,) array of per-camera data-fit
        weights (see `cost_and_grad`), in `self.cam_keys` order. None (the
        default) is uniform weighting -- original behavior.
        """
        # CUPY_AVAILABLE only reflects whether `import cupy` succeeded, not
        # whether a usable CUDA device is actually present -- cupy is
        # importable on CPU-only nodes too (it's a normal env dependency),
        # but cp.asarray() then fails at runtime (e.g. CUDARuntimeError:
        # insufficient driver) if there's no GPU. Fall back to NumPy on any
        # such runtime failure instead of crashing, so the same code works
        # unchanged on both GPU and CPU-only queues.
        use_gpu_backend = CUPY_AVAILABLE and not avoid_gpu
        if use_gpu_backend:
            try:
                self._xp = cp
                self._A = cp.asarray(self.A)
                self._A_3d = cp.asarray(self.A_3d)
                self._A_mean = cp.asarray(self.A_mean)
                self._obs_norm = cp.asarray(self.obs_norm)
                self._normals = cp.asarray(self.normals)
            except Exception as e:
                print(
                    f"CuPy is installed but no usable GPU was found ({e!r}) — falling back to NumPy CPU backend"
                )
                use_gpu_backend = False
        if not use_gpu_backend:
            self._xp = np
            self._A = self.A
            self._A_3d = self.A_3d
            self._A_mean = self.A_mean
            self._obs_norm = self.obs_norm
            self._normals = self.normals
            if not CUPY_AVAILABLE:
                print("CuPy unavailable — using NumPy CPU backend")

        _use_gpu = self._xp is not np

        if parametrization not in ("log", "direct"):
            raise ValueError(
                f"parametrization must be 'log' or 'direct', got {parametrization!r}"
            )
        if parametrization == "direct" and optimizer != "lbfgs":
            print(
                f"parametrization='direct' needs a bounded method — overriding optimizer={optimizer!r} -> 'lbfgs'"
            )
            optimizer = "lbfgs"

        if optimizer not in ["lbfgs", "cg"]:
            raise ValueError(
                f"Invalid optimizer '{optimizer}'. Choose 'lbfgs' or 'cg'."
            )
        sci_method = {"lbfgs": "L-BFGS-B", "cg": "CG"}[optimizer]

        print(
            f"Running {sci_method}  (n_params={self.n_facets}, parametrization={parametrization!r})..."
        )

        def func(a):
            return self.cost_and_grad(
                a=a,
                lambda_s=convexity_regularization,
                _use_gpu=_use_gpu,
                lambda_a=scale_penalty,
                parametrization=parametrization,
                cam_weights=cam_weights,
                l1_regularization=l1_regularization,
            )

        _iter = [0]

        def _cb(a):
            _iter[0] += 1
            if _iter[0] % 1000 == 0:
                c, _ = func(a)
                print(f"  iter {_iter[0]:4d}: cost = {c:.6f}")

        if parametrization == "log":
            a_init = np.log(self.areas_init)
            bounds = None
        else:
            a_init = self.areas_init.copy()
            floor = (
                direct_area_floor
                if direct_area_floor is not None
                else 1e-6 * self.areas_init.mean()
            )
            bounds = [(floor, None)] * self.n_facets

        result = minimize(
            func,
            a_init,
            jac=True,
            method=sci_method,
            bounds=bounds,
            callback=_cb,
            options={"maxiter": max_iter, "gtol": gtol},
        )
        self.a_opt = result.x
        self.areas_opt = np.exp(self.a_opt) if parametrization == "log" else self.a_opt
        print(f"\n{result.message}")
        print(f"Cost: {result.fun:.6f}")
        return

    def fit_areas_joint(
        self,
        convexity_regularization=0.1,
        optimizer="cg",
        scale_penalty=None,
        max_iter=10000,
        gtol=1e-8,
        avoid_gpu=False,
        parametrization="log",
        direct_area_floor=None,
        cam_weights=None,
        channel_weights=None,
    ):
        """
        Joint-mode analogue of `fit_areas`: optimizes ONE shared area
        vector against BOTH intensity and binary lightcurve data at once
        (see `cost_and_grad_joint`).
        """
        if not hasattr(self, "channels"):
            raise ValueError(
                "Joint lightcurve data not loaded. Call load_lightcurve_data_joint() first."
            )

        use_gpu_backend = CUPY_AVAILABLE and not avoid_gpu
        if use_gpu_backend:
            try:
                self._xp = cp
                self._normals = cp.asarray(self.normals)
                for channel in self.channels:
                    channel["_A"] = cp.asarray(channel["A"])
                    channel["_A_mean"] = cp.asarray(channel["A_mean"])
                    channel["_obs_norm"] = cp.asarray(channel["obs_norm"])
            except Exception as e:
                print(
                    f"CuPy is installed but no usable GPU was found ({e!r}) — falling back to NumPy CPU backend"
                )
                use_gpu_backend = False
        if not use_gpu_backend:
            self._xp = np
            self._normals = self.normals
            for channel in self.channels:
                channel["_A"] = channel["A"]
                channel["_A_mean"] = channel["A_mean"]
                channel["_obs_norm"] = channel["obs_norm"]
            if not CUPY_AVAILABLE:
                print("CuPy unavailable — using NumPy CPU backend")

        _use_gpu = self._xp is not np

        if parametrization not in ("log", "direct"):
            raise ValueError(
                f"parametrization must be 'log' or 'direct', got {parametrization!r}"
            )
        if parametrization == "direct" and optimizer != "lbfgs":
            print(
                f"parametrization='direct' needs a bounded method — overriding optimizer={optimizer!r} -> 'lbfgs'"
            )
            optimizer = "lbfgs"

        if optimizer not in ["lbfgs", "cg"]:
            raise ValueError(
                f"Invalid optimizer '{optimizer}'. Choose 'lbfgs' or 'cg'."
            )
        sci_method = {"lbfgs": "L-BFGS-B", "cg": "CG"}[optimizer]

        print(
            f"Running {sci_method}  (n_params={self.n_facets}, parametrization={parametrization!r}, joint intensity+binary)..."
        )

        def func(a):
            return self.cost_and_grad_joint(
                a=a,
                lambda_s=convexity_regularization,
                _use_gpu=_use_gpu,
                lambda_a=scale_penalty,
                parametrization=parametrization,
                cam_weights=cam_weights,
                channel_weights=channel_weights,
            )

        _iter = [0]

        def _cb(a):
            _iter[0] += 1
            if _iter[0] % 1000 == 0 or _iter[0] == 1:
                c, _ = func(a)
                print(f"  iter {_iter[0]:4d}: cost = {c:.6f}")

        if parametrization == "log":
            a_init = np.log(self.areas_init)
            bounds = None
        else:
            a_init = self.areas_init.copy()
            floor = (
                direct_area_floor
                if direct_area_floor is not None
                else 1e-6 * self.areas_init.mean()
            )
            bounds = [(floor, None)] * self.n_facets

        result = minimize(
            func,
            a_init,
            jac=True,
            method=sci_method,
            bounds=bounds,
            callback=_cb,
            options={"maxiter": max_iter, "gtol": gtol},
        )
        self.a_opt = result.x
        self.areas_opt = np.exp(self.a_opt) if parametrization == "log" else self.a_opt
        print(f"\n{result.message}")
        print(f"Cost: {result.fun:.6f}")
        return

    def lightcurve_fit(self, show=True):
        """
        Compare the observed normalized lightcurve data against the fitted
        model's prediction: plots (if `show`) and returns a per-camera
        DataFrame of MSE, correlation and RRMSE.
        """
        Ag_opt = (self.A @ self.areas_opt).reshape(self.n_cams, self.n_phases)
        pred_norm = Ag_opt / Ag_opt.mean(axis=1, keepdims=True)
        phase_axis = np.linspace(0, 1, self.n_phases, endpoint=False)

        rows = []
        for i, cam in enumerate(self.cam_keys):
            obs, pred = self.obs_norm[i], pred_norm[i]
            rows.append(
                {
                    "camera": cam,
                    "mse": float(np.mean((obs - pred) ** 2)),
                    "rrmse": _rrmse(pred, obs),
                    "corr": _corr(pred, obs),
                }
            )
        df = pd.DataFrame(rows).set_index("camera")
        self.lightcurve_metrics = df

        if show:
            n_cols = 3
            n_rows = int(np.ceil(self.n_cams / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True
            )
            axes = np.atleast_1d(axes).ravel()
            for ax, cam, i in zip(axes, self.cam_keys, range(self.n_cams)):
                ax.plot(
                    phase_axis,
                    self.obs_norm[i],
                    lw=1.2,
                    color="steelblue",
                    label=self.origin,
                )
                ax.plot(
                    phase_axis,
                    pred_norm[i],
                    lw=1.2,
                    color="tomato",
                    label="inversion",
                    ls="--",
                )
                ax.set_title(
                    f"{cam}  (corr={df.loc[cam, 'corr']:.2f}, rrmse={df.loc[cam, 'rrmse']:.2f})",
                    fontsize=8,
                )
                ax.set_yticks([])
            for ax in axes[self.n_cams :]:
                ax.axis("off")
            axes[0].legend(fontsize=7)
            fig.suptitle(
                f"Asteroid {self.asteroid} ({self.origin}/{self.curve_type}) — lightcurve fit",
                y=1.002,
            )
            fig.tight_layout()
            plt.show()

        print(f"Overall MSE (normalised): {df['mse'].mean():.6f}")
        print(f"Overall RRMSE:            {df['rrmse'].mean():.6f}")
        print(f"Overall correlation:      {df['corr'].mean():.6f}")
        return df

    def lightcurve_fit_joint(self, show=True):
        """
        Joint-mode analogue of `lightcurve_fit`: computes per-camera fit
        metrics (mse/rrmse/corr) separately for each channel (intensity,
        binary in self.channels) against the single shared self.areas_opt
        from `fit_areas_joint`. Returns {curve_type: DataFrame}, also
        cached as self.lightcurve_metrics_joint.
        """
        if not hasattr(self, "channels"):
            raise ValueError("Joint lightcurve data not loaded.")
        dfs = {}
        for channel in self.channels:
            ct = channel["curve_type"]
            A, obs_norm, n_phases = (
                channel["A"],
                channel["obs_norm"],
                channel["n_phases"],
            )
            Ag_opt = (A @ self.areas_opt).reshape(self.n_cams, n_phases)
            pred_norm = Ag_opt / Ag_opt.mean(axis=1, keepdims=True)
            phase_axis = np.linspace(0, 1, n_phases, endpoint=False)

            rows = []
            for i, cam in enumerate(self.cam_keys):
                obs, pred = obs_norm[i], pred_norm[i]
                rows.append(
                    {
                        "camera": cam,
                        "mse": float(np.mean((obs - pred) ** 2)),
                        "rrmse": _rrmse(pred, obs),
                        "corr": _corr(pred, obs),
                    }
                )
            df = pd.DataFrame(rows).set_index("camera")
            dfs[ct] = df

            if show:
                n_cols = 3
                n_rows = int(np.ceil(self.n_cams / n_cols))
                fig, axes = plt.subplots(
                    n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True
                )
                axes = np.atleast_1d(axes).ravel()
                for ax, cam, i in zip(axes, self.cam_keys, range(self.n_cams)):
                    ax.plot(
                        phase_axis,
                        obs_norm[i],
                        lw=1.2,
                        color="steelblue",
                        label=self.origin,
                    )
                    ax.plot(
                        phase_axis,
                        pred_norm[i],
                        lw=1.2,
                        color="tomato",
                        label="inversion",
                        ls="--",
                    )
                    ax.set_title(
                        f"{cam}  (corr={df.loc[cam, 'corr']:.2f}, rrmse={df.loc[cam, 'rrmse']:.2f})",
                        fontsize=8,
                    )
                    ax.set_yticks([])
                for ax in axes[self.n_cams :]:
                    ax.axis("off")
                axes[0].legend(fontsize=7)
                fig.suptitle(
                    f"Asteroid {self.asteroid} ({self.origin}/{ct}, joint fit) — lightcurve fit",
                    y=1.002,
                )
                fig.tight_layout()
                plt.show()

            print(f"[{ct}] Overall MSE (normalised): {df['mse'].mean():.6f}")
            print(f"[{ct}] Overall RRMSE:            {df['rrmse'].mean():.6f}")
            print(f"[{ct}] Overall correlation:      {df['corr'].mean():.6f}")

        self.lightcurve_metrics_joint = dfs
        return dfs

    def refine_areas(self, small_tolerance=None):
        areas_refined = self.areas_opt.copy()
        normals_refined = self.normals.copy()
        if small_tolerance is not None:
            mean_area = np.mean(areas_refined)
            keep_mask = areas_refined > (small_tolerance * mean_area)
            areas_refined = areas_refined[keep_mask]
            normals_refined = normals_refined[keep_mask]
            print(
                f"Removed {np.sum(~keep_mask)} small areas below {small_tolerance * 100:.2f}% of mean area."
            )
        total_area = np.sum(areas_refined)

        closure = normals_refined.T @ areas_refined
        dark_area = np.linalg.norm(closure)
        if dark_area > total_area * 1e-10:
            areas_refined = np.concatenate([areas_refined, np.array([dark_area])])
            normals_refined = np.vstack([normals_refined, -closure / dark_area])
            self.dark_area_percentage = 100 * dark_area / (total_area + dark_area)
            print(
                f"Added dark area. Dark area percentage: {self.dark_area_percentage:.6f}%.\n"
                "If this is large (>1%), recompute the areas with larger convexity regularization."
            )
        else:
            self.dark_area_percentage = 0.0
            print("No significant dark area needed.")

        self.areas_refined = areas_refined
        self.normals_refined = normals_refined
        return

    def minkowski_reconstruction(self, max_iter=6000, epsilon=0.02):
        """
        `max_iter`/`epsilon` are passed straight through to
        `minkowski_reconstruct` (defaults there: max_iter=3000,
        epsilon=0.005).
        """
        if not hasattr(self, "areas_refined") or not hasattr(self, "normals_refined"):
            raise ValueError(
                "Refined areas and normals not computed. Call refine_areas() first."
            )

        vertices_cloud = minkowski_reconstruct(
            self.areas_refined,
            self.normals_refined,
            verbose=True,
            epsilon=epsilon,
            max_iter=max_iter,
        )
        mesh_verts, mesh_faces = build_convex_mesh(vertices_cloud)
        print(
            f"Reconstructed convex mesh: {len(mesh_verts)} vertices, {len(mesh_faces)} faces"
        )

        vx = mesh_verts[:, 0]
        vy = mesh_verts[:, 1]
        max_radius = np.sqrt(vx**2 + vy**2).max()

        self.recon_vertices = mesh_verts * (self.cylinder_radius / max_radius)
        self.recon_faces = mesh_faces
        return self.recon_vertices, self.recon_faces

    def plot_reconstruction(self, show=True):
        if not hasattr(self, "recon_vertices") or not hasattr(self, "recon_faces"):
            raise ValueError(
                "Reconstructed mesh not available. Call minkowski_reconstruction() first."
            )

        mesh = pv.PolyData(
            self.recon_vertices,
            np.hstack([np.full((len(self.recon_faces), 1), 3), self.recon_faces]),
        )
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color="lightgray", show_edges=True)
        plotter.add_axes()
        plotter.show_grid()
        if show:
            plotter.show()
        return plotter

    def _plot_multi_view(self, out_path, views=None, show=False):
        """
        Static multi-angle preview of the reconstruction

        Rows (top to bottom):
        1. Reconstruction, raw
        2. Reconstruction, cylinder-normalized
        3. Ground truth
        4. Ground truth's convex hull
        """
        views = views or DEFAULT_RECON_VIEWS
        has_truth = hasattr(self, "ground_truth_mesh_vertices")
        lim = self.cylinder_radius * 1.1

        recon_mesh_raw = trimesh.Trimesh(
            vertices=self.recon_vertices, faces=self.recon_faces, process=True
        )
        recon_mesh_norm = normalize_mesh_to_challenge_cylinder(
            recon_mesh_raw, target_R=self.cylinder_radius
        )

        rows = [
            ("Reconstruction (raw)", self.recon_vertices, self.recon_faces),
            (
                "Reconstruction (cylinder-normalized)",
                np.asarray(recon_mesh_norm.vertices),
                np.asarray(recon_mesh_norm.faces),
            ),
        ]
        if has_truth:
            rows.append(
                (
                    "Ground truth",
                    self.ground_truth_mesh_vertices,
                    self.ground_truth_mesh_faces,
                )
            )
            gt_mesh = trimesh.Trimesh(
                vertices=self.ground_truth_mesh_vertices,
                faces=self.ground_truth_mesh_faces,
                process=True,
            )
            gt_hull = gt_mesh.convex_hull
            rows.append(
                (
                    "Ground truth convex hull",
                    np.asarray(gt_hull.vertices),
                    np.asarray(gt_hull.faces),
                )
            )
        n_rows = len(rows)

        fig = plt.figure(figsize=(4 * len(views), 4.5 * n_rows))
        for r, (row_label, verts, faces) in enumerate(rows):
            tris = verts[faces]  # (n_faces, 3, 3)
            for i, (label, az, elev) in enumerate(views):
                ax = fig.add_subplot(
                    n_rows, len(views), r * len(views) + i + 1, projection="3d"
                )
                ax.add_collection3d(
                    Poly3DCollection(
                        tris, facecolor="lightgray", edgecolor="k", linewidths=0.1
                    )
                )
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-lim, lim)
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=elev, azim=az)
                ax.set_title(f"{row_label}: {label}", fontsize=9)
                ax.set_axis_off()

        fig.suptitle(
            f"Asteroid {self.asteroid} reconstruction ({self.origin}/{self.curve_type})",
            y=0.995,
        )
        fig.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out_path

    def _plot_lightcurve_comparison(self, out_path, show=False):
        """
        Per-camera plot of the observed lightcurve data against two
        forward-modeled curves: the reconstruction's fit, and the
        ground-truth mesh's own forward simulation (both computed by
        `evaluate_against_ground_truth`, which must be called first --
        this only reads its cached `self.ground_truth_curves`). All three
        are on the same phase axis (the ground-truth curve has already
        been circularly shifted into that frame, see
        `evaluate_against_ground_truth`).
        """
        if not hasattr(self, "ground_truth_curves"):
            raise ValueError(
                "No ground-truth curves cached. Call evaluate_against_ground_truth() first."
            )

        recon = self.ground_truth_curves["recon"]
        true = self.ground_truth_curves["true"]
        phase_axis = np.linspace(0, 1, self.n_phases, endpoint=False)

        n_cols = 3
        n_rows = int(np.ceil(self.n_cams / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, cam, i in zip(axes, self.cam_keys, range(self.n_cams)):
            ax.plot(
                phase_axis,
                self.obs_norm[i],
                lw=1.2,
                color="steelblue",
                label=f"{self.origin} data",
            )
            ax.plot(
                phase_axis,
                recon[cam],
                lw=1.2,
                color="tomato",
                ls="--",
                label="reconstruction (fwd model)",
            )
            ax.plot(
                phase_axis,
                true[cam],
                lw=1.2,
                color="seagreen",
                ls=":",
                label="ground-truth mesh (fwd model)",
            )
            ax.set_title(cam, fontsize=8)
            ax.set_yticks([])
        for ax in axes[self.n_cams :]:
            ax.axis("off")
        axes[0].legend(fontsize=7)
        fig.suptitle(
            f"Asteroid {self.asteroid} ({self.origin}/{self.curve_type}) — data vs. reconstruction vs. ground truth",
            y=1.002,
        )
        fig.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out_path

    def _plot_lightcurve_comparison_joint(self, out_dir, name, show=False):
        """
        Joint-mode analogue of `_plot_lightcurve_comparison`: one plot per
        channel (intensity, binary), since each has its own phase grid and
        scattering law. Reads `self.ground_truth_curves_joint` (from
        `evaluate_against_ground_truth_joint`). Returns {curve_type: path}.
        """
        if not hasattr(self, "ground_truth_curves_joint"):
            raise ValueError(
                "No ground-truth curves cached. Call evaluate_against_ground_truth_joint() first."
            )

        out_dir = Path(out_dir)
        paths = {}
        for channel in self.channels:
            ct = channel["curve_type"]
            if ct not in self.ground_truth_curves_joint:
                continue
            cache = self.ground_truth_curves_joint[ct]
            recon, true = cache["recon"], cache["true"]
            n_phases = channel["n_phases"]
            obs_norm = channel["obs_norm"]
            phase_axis = np.linspace(0, 1, n_phases, endpoint=False)

            n_cols = 3
            n_rows = int(np.ceil(self.n_cams / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True
            )
            axes = np.atleast_1d(axes).ravel()
            for ax, cam, i in zip(axes, self.cam_keys, range(self.n_cams)):
                ax.plot(
                    phase_axis,
                    obs_norm[i],
                    lw=1.2,
                    color="steelblue",
                    label=f"{self.origin} data",
                )
                ax.plot(
                    phase_axis,
                    recon[cam],
                    lw=1.2,
                    color="tomato",
                    ls="--",
                    label="reconstruction (fwd model)",
                )
                ax.plot(
                    phase_axis,
                    true[cam],
                    lw=1.2,
                    color="seagreen",
                    ls=":",
                    label="ground-truth mesh (fwd model)",
                )
                ax.set_title(cam, fontsize=8)
                ax.set_yticks([])
            for ax in axes[self.n_cams :]:
                ax.axis("off")
            axes[0].legend(fontsize=7)
            fig.suptitle(
                f"Asteroid {self.asteroid} ({self.origin}/{ct}, joint fit) — data vs. reconstruction vs. ground truth",
                y=1.002,
            )
            fig.tight_layout()
            out_path = out_dir / f"{name}_{ct}_lightcurves_vs_truth.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)
            paths[ct] = out_path
        return paths

    def save_reconstruction(
        self,
        out_dir="reconstructions",
        name=None,
        stl=True,
        views=True,
        lightcurves=True,
        view_list=None,
        show=False,
    ):
        """
        Save the current reconstruction (`self.recon_vertices`/
        `recon_faces`, from `minkowski_reconstruction()`) to disk:
        - an STL mesh file, via `save_stl`.
        - a static multi-angle preview PNG (see `_plot_multi_view`).
        - if `evaluate_against_ground_truth()` was already called (so
          ground-truth curves are cached), a plot comparing observed data,
          the reconstruction's fit, and the ground-truth mesh's own
          forward-simulated curves per camera (see
          `_plot_lightcurve_comparison`). Silently skipped otherwise.

        Filenames default to f"asteroid{self.asteroid}_{self.origin}_{self.curve_type}"
        under `out_dir` (created if missing) so different data-kind runs on
        the same asteroid don't overwrite each other.
        """
        if not hasattr(self, "recon_vertices"):
            raise ValueError(
                "No reconstruction available. Call minkowski_reconstruction() first."
            )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = name or f"asteroid{self.asteroid}_{self.origin}_{self.curve_type}"

        paths = {}
        if stl:
            stl_path = out_dir / f"{name}.stl"
            save_stl(self.recon_vertices, self.recon_faces, stl_path)
            paths["stl"] = stl_path
            print(f"Saved reconstruction STL to {stl_path}")
        if views:
            views_path = self._plot_multi_view(
                out_dir / f"{name}_views.png", views=view_list, show=show
            )
            paths["views_png"] = views_path
            print(f"Saved multi-angle view plot to {views_path}")
        if lightcurves:
            if hasattr(self, "ground_truth_curves_joint"):
                lc_paths = self._plot_lightcurve_comparison_joint(
                    out_dir, name, show=show
                )
                paths["lightcurves_png"] = lc_paths
                for ct, p in lc_paths.items():
                    print(f"Saved {ct} lightcurve comparison plot to {p}")
            elif hasattr(self, "ground_truth_curves"):
                lc_path = self._plot_lightcurve_comparison(
                    out_dir / f"{name}_lightcurves_vs_truth.png", show=show
                )
                paths["lightcurves_png"] = lc_path
                print(f"Saved lightcurve comparison plot to {lc_path}")
            else:
                print(
                    "No ground-truth curves cached (call evaluate_against_ground_truth() first) — skipping lightcurve comparison plot."
                )
        return paths

    def evaluate_against_ground_truth(
        self,
        pitch=0.02,
        align_phase=True,
        phase_batch_size=64,
        ground_truth_stl_path=None,
        compute_voxel_similarity=True,
        group_fn=_cam_group,
    ):
        """
        Compare the current reconstruction against the public ground-truth
        STL for `self.asteroid` (asteroids 1-3 only)

        `ground_truth_stl_path` overrides the official full-resolution STL
        (`get_model_stl_path`) with an alternate mesh -- e.g. asteroid 2's
        16-facet `stls/asteroid2_simplified.stl`

        `compute_voxel_similarity`: voxelizing/comparing meshes (below) is
        the most expensive part of this method for a larger ground-truth
        mesh (e.g. asteroid 1's 24k-facet convex hull). Set False to skip it
        and only get the (cheaper) lightcurve-space comparison.

        - `calculate_voxel_measure` (mesh-space similarity in
          [0, 1]) between the reconstructed mesh and the ground-truth STL.
          Both are passed through `normalize_mesh_to_challenge_cylinder`
          with the same `target_R=self.cylinder_radius` so they're on
          identical scale/centering conventions
        - Lightcurve correlation/RRMSE, per camera and averaged, between
          the reconstruction's fitted prediction and curves
          forward-simulated from the ground-truth mesh using the same
          camera geometry and fitted scattering-law parameters. Unlike the
          mesh comparison, this **is** phase-aligned (`align_phase`)

        Caches per-camera curves (`self.ground_truth_curves`) for
        `save_reconstruction`'s lightcurve comparison plot.
        """
        if ground_truth_stl_path is None and self.asteroid > 3:
            print(
                f"No public ground-truth STL for asteroid {self.asteroid} (only 1-3 have one); skipping."
            )
            return None
        if not hasattr(self, "recon_vertices"):
            raise ValueError(
                "No reconstruction available. Call minkowski_reconstruction() first."
            )

        stl_path = (
            ground_truth_stl_path
            if ground_truth_stl_path is not None
            else get_model_stl_path(self.asteroid)
        )
        true_mesh_raw = trimesh.load(stl_path, force="mesh")
        true_mesh = normalize_mesh_to_challenge_cylinder(
            true_mesh_raw, target_R=self.cylinder_radius
        )
        # cached for save_reconstruction's multi-view plot's ground-truth row
        self.ground_truth_mesh_vertices = np.asarray(true_mesh.vertices)
        self.ground_truth_mesh_faces = np.asarray(true_mesh.faces)

        voxel_score = None
        voxel_score_solid = None
        if compute_voxel_similarity:
            pred_mesh_raw = trimesh.Trimesh(
                vertices=self.recon_vertices, faces=self.recon_faces, process=True
            )
            pred_mesh = normalize_mesh_to_challenge_cylinder(
                pred_mesh_raw, target_R=self.cylinder_radius
            )
            voxel_score = float(
                calculate_voxel_measure(true_mesh, pred_mesh, pitch=pitch)
            )
            self.voxel_similarity = voxel_score
            print(
                f"Voxel similarity vs. ground truth (surface-shell): {voxel_score:.4f}"
            )
            voxel_score_solid = _calculate_voxel_measure_solid(
                true_mesh, pred_mesh, pitch=pitch
            )
            self.voxel_similarity_solid = voxel_score_solid
            print(
                f"Voxel similarity vs. ground truth (solid-fill):      {voxel_score_solid:.4f}"
            )
        else:
            print("Skipping voxel similarity (compute_voxel_similarity=False).")

        # --- lightcurve comparison: forward-simulate the ground-truth mesh
        # with the same camera geometry & fitted scattering parameters.
        # Ground-truth STLs are full-resolution (e.g. asteroid 2's ~538k
        # facets, asteroid 1's ~800k) -- computed phase-batched (see
        # `_simulate_curves_for_shape`) so peak memory stays bounded
        # regardless of facet count or n_phases (842 for real data). ---
        true_normals, true_areas = _facet_normals_and_areas(true_mesh)
        params = getattr(self, "scattering_params_used", None) or getattr(
            self, "scattering_params", None
        )
        true_curves = self._simulate_curves_for_shape(
            true_normals,
            true_areas,
            params=params,
            default=0.1,
            phase_batch_size=phase_batch_size,
            group_fn=group_fn,
        )

        Ag_opt = (self.A @ self.areas_opt).reshape(self.n_cams, self.n_phases)
        recon_norm = Ag_opt / Ag_opt.mean(axis=1, keepdims=True)
        # "native" frame -- directly aligned with self.obs_norm, since both
        # were jointly fit together (no shift needed between them).
        recon_curves_native = {
            cam: recon_norm[i] for i, cam in enumerate(self.cam_keys)
        }

        shift = 0
        true_curves_native = true_curves
        if align_phase:
            # _circular_align_all(a, b) finds k s.t. roll(a, k) best matches
            # b, i.e. roll(recon, k) ~= true. Equivalently
            # recon ~= roll(true, -k), so roll the true-mesh curves the
            # other way to bring them into the native (obs/recon) frame
            # instead -- keeps everything on one consistent axis for both
            # the metrics below and the 3-curve comparison plot.
            _, shift = _circular_align_all(recon_curves_native, true_curves)
            true_curves_native = {
                cam: np.roll(true_curves[cam], -shift) for cam in true_curves
            }

        rows = [
            {
                "camera": cam,
                "corr": _corr(recon_curves_native[cam], true_curves_native[cam]),
                "rrmse": _rrmse(recon_curves_native[cam], true_curves_native[cam]),
            }
            for cam in self.cam_keys
        ]
        df = pd.DataFrame(rows).set_index("camera")
        self.ground_truth_lightcurve_metrics = df
        self.ground_truth_phase_shift = shift
        self.ground_truth_curves = {
            "recon": recon_curves_native,
            "true": true_curves_native,
            "phase_shift": shift,
        }

        print(f"Lightcurve vs. ground-truth mesh (phase shift={shift} samples):")
        print(f"  mean correlation: {df['corr'].mean():.4f}")
        print(f"  mean RRMSE:       {df['rrmse'].mean():.4f}")

        return {
            "voxel_similarity": voxel_score,
            "voxel_similarity_solid": voxel_score_solid,
            "lightcurve_corr_mean": float(df["corr"].mean()),
            "lightcurve_rrmse_mean": float(df["rrmse"].mean()),
            "lightcurve_phase_shift": shift,
            "lightcurve_metrics_per_camera": df,
        }

    def evaluate_against_ground_truth_joint(
        self,
        pitch=0.02,
        align_phase=True,
        phase_batch_size=64,
        ground_truth_stl_path=None,
        compute_voxel_similarity=True,
        group_fn=_cam_group,
    ):
        """
        Joint-mode analogue of `evaluate_against_ground_truth`.
        """
        if ground_truth_stl_path is None and self.asteroid > 3:
            print(
                f"No public ground-truth STL for asteroid {self.asteroid} (only 1-3 have one); skipping."
            )
            return None
        if not hasattr(self, "recon_vertices"):
            raise ValueError(
                "No reconstruction available. Call minkowski_reconstruction() first."
            )
        if not hasattr(self, "channels"):
            raise ValueError(
                "Joint lightcurve data not loaded. Call load_lightcurve_data_joint() first."
            )

        stl_path = (
            ground_truth_stl_path
            if ground_truth_stl_path is not None
            else get_model_stl_path(self.asteroid)
        )
        true_mesh_raw = trimesh.load(stl_path, force="mesh")
        true_mesh = normalize_mesh_to_challenge_cylinder(
            true_mesh_raw, target_R=self.cylinder_radius
        )
        self.ground_truth_mesh_vertices = np.asarray(true_mesh.vertices)
        self.ground_truth_mesh_faces = np.asarray(true_mesh.faces)

        voxel_score = None
        voxel_score_solid = None
        if compute_voxel_similarity:
            pred_mesh_raw = trimesh.Trimesh(
                vertices=self.recon_vertices, faces=self.recon_faces, process=True
            )
            pred_mesh = normalize_mesh_to_challenge_cylinder(
                pred_mesh_raw, target_R=self.cylinder_radius
            )
            voxel_score = float(
                calculate_voxel_measure(true_mesh, pred_mesh, pitch=pitch)
            )
            self.voxel_similarity = voxel_score
            print(
                f"Voxel similarity vs. ground truth (surface-shell, official metric): {voxel_score:.4f}"
            )
            voxel_score_solid = _calculate_voxel_measure_solid(
                true_mesh, pred_mesh, pitch=pitch
            )
            self.voxel_similarity_solid = voxel_score_solid
            print(
                f"Voxel similarity vs. ground truth (solid-fill, supplementary):      {voxel_score_solid:.4f}"
            )
        else:
            print("Skipping voxel similarity (compute_voxel_similarity=False).")

        true_normals, true_areas = _facet_normals_and_areas(true_mesh)

        results = {
            "voxel_similarity": voxel_score,
            "voxel_similarity_solid": voxel_score_solid,
        }
        self.ground_truth_curves_joint = {}
        for channel in self.channels:
            ct = channel["curve_type"]
            params = channel.get("scattering_params_used") or channel.get(
                "scattering_params"
            )
            true_curves = self._simulate_curves_for_shape(
                true_normals,
                true_areas,
                params=params,
                default=0.1,
                phase_batch_size=phase_batch_size,
                group_fn=group_fn,
                curve_type=ct,
                phases=channel["phases"],
                n_phases=channel["n_phases"],
            )

            A, n_phases = channel["A"], channel["n_phases"]
            Ag_opt = (A @ self.areas_opt).reshape(self.n_cams, n_phases)
            recon_norm = Ag_opt / Ag_opt.mean(axis=1, keepdims=True)
            recon_curves_native = {
                cam: recon_norm[i] for i, cam in enumerate(self.cam_keys)
            }

            shift = 0
            true_curves_native = true_curves
            if align_phase:
                _, shift = _circular_align_all(recon_curves_native, true_curves)
                true_curves_native = {
                    cam: np.roll(true_curves[cam], -shift) for cam in true_curves
                }

            rows = [
                {
                    "camera": cam,
                    "corr": _corr(recon_curves_native[cam], true_curves_native[cam]),
                    "rrmse": _rrmse(recon_curves_native[cam], true_curves_native[cam]),
                }
                for cam in self.cam_keys
            ]
            df = pd.DataFrame(rows).set_index("camera")

            self.ground_truth_curves_joint[ct] = {
                "recon": recon_curves_native,
                "true": true_curves_native,
                "phase_shift": shift,
            }

            print(
                f"[{ct}] Lightcurve vs. ground-truth mesh (phase shift={shift} samples):"
            )
            print(f"  mean correlation: {df['corr'].mean():.4f}")
            print(f"  mean RRMSE:       {df['rrmse'].mean():.4f}")

            results[ct] = {
                "lightcurve_corr_mean": float(df["corr"].mean()),
                "lightcurve_rrmse_mean": float(df["rrmse"].mean()),
                "lightcurve_phase_shift": shift,
                "lightcurve_metrics_per_camera": df,
            }

        self.ground_truth_lightcurve_metrics_joint = results
        return results
