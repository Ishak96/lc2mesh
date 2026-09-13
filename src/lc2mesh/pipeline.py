"""Differentiable mesh -> lightcurve pipeline (convex, or self-occluding).

Transcribed unchanged from the reference notebook: the convex LS+Lambert matrix
``A[C, P, F]`` is optionally multiplied by a ray-cast sun/observer visibility
mask. The mask is piecewise-constant (recomputed under ``no_grad`` every
``occlusion_refresh_every`` calls) so gradients still flow through ``A`` and the
facet areas. Stage 1 keeps it OFF (convex-hull prior); Stage 2 turns it ON so
concavities become observable.
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from lc2mesh.forward_model import AsteroidForwardModel


def facet_geometry(
    vertices: torch.Tensor, faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Facet normals [F,3] and areas [F] from vertices [V,3] (differentiable)."""
    tri = vertices[faces]
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    cross = torch.cross(edge1, edge2, dim=1)
    twice_area = cross.norm(dim=1)
    normals = cross / twice_area.clamp_min(1e-12).unsqueeze(1)
    return normals, 0.5 * twice_area


class LightcurvePipeline:
    """Callable ``vertices [V, 3] -> normalized lightcurves [C, P]``."""

    def __init__(
        self,
        *,
        faces: torch.Tensor,
        phases: torch.Tensor,
        e_sun: torch.Tensor,
        e_obs: torch.Tensor,
        device: torch.device,
        lambert_weight: float,
        occlusion_enabled: bool,
        occlusion_refresh_every: int,
        occlusion_n_phases: int,
        occlusion_eps: float,
    ) -> None:
        self.faces = faces
        self.faces_np = faces.detach().cpu().numpy()
        self.phases = phases
        self.e_sun = e_sun
        self.e_obs = e_obs
        self.device = device
        self.lambert_weight = float(lambert_weight)
        self.occlusion_enabled = bool(occlusion_enabled)
        self.occlusion_refresh_every = int(occlusion_refresh_every)
        self.occlusion_eps = float(occlusion_eps)

        n_phase = int(phases.shape[0])
        self._coarse_idx = torch.linspace(
            0, n_phase - 1, min(occlusion_n_phases, n_phase)
        ).round().long()
        self._phase_slot = (
            torch.arange(n_phase, dtype=torch.float32)
            / max(n_phase - 1, 1)
            * (self._coarse_idx.numel() - 1)
        ).round().long().to(device)

        self.state = {"active": False, "mask": None, "calls": 0}

    def reset_occlusion(self, active: bool) -> None:
        """Switch the occlusion physics on/off and drop any cached mask."""
        self.state["active"] = bool(active)
        self.state["mask"] = None
        self.state["calls"] = 0

    def facet_geometry(self, vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return facet_geometry(vertices, self.faces)

    @torch.no_grad()
    def compute_visibility_mask(self, vertices: torch.Tensor) -> torch.Tensor:
        """Per-(camera, phase, facet) visibility in {0,1}, shape [C, P, F].

        A facet is visible for camera c at phase p if it is neither self-shadowed
        (ray toward the sun is unobstructed) nor occluded (ray toward the observer
        is unobstructed). Directions use the same +z phase rotation as the forward
        matrix. Rays are cast on a coarse phase grid and nearest-upsampled to P.
        """
        verts_np = vertices.detach().cpu().numpy()
        tri_mesh = trimesh.Trimesh(vertices=verts_np, faces=self.faces_np, process=False)
        intersector = RayMeshIntersector(tri_mesh)
        origins = tri_mesh.triangles_center + self.occlusion_eps * tri_mesh.face_normals
        n_facets = origins.shape[0]
        phases_coarse = self.phases[self._coarse_idx]
        e_sun_rot = AsteroidForwardModel.rotate_vectors_z(self.e_sun, phases_coarse).cpu().numpy()  # [Pm,3]
        e_obs_rot = AsteroidForwardModel.rotate_vectors_z(self.e_obs, phases_coarse).cpu().numpy()  # [Pm,C,3]
        n_coarse, n_cams = e_sun_rot.shape[0], e_obs_rot.shape[1]
        origins_tiled = np.tile(origins, (n_coarse, 1))
        lit = ~intersector.intersects_any(
            origins_tiled, np.repeat(e_sun_rot, n_facets, axis=0)
        ).reshape(n_coarse, n_facets)
        visible = np.empty((n_cams, n_coarse, n_facets), dtype=bool)
        for cam in range(n_cams):
            seen = ~intersector.intersects_any(
                origins_tiled, np.repeat(e_obs_rot[:, cam, :], n_facets, axis=0)
            ).reshape(n_coarse, n_facets)
            visible[cam] = seen & lit
        mask_coarse = torch.as_tensor(visible, dtype=torch.float32, device=self.device)  # [C,Pm,F]
        return mask_coarse[:, self._phase_slot, :]  # [C,P,F]

    def predicted_lightcurves(self, vertices: torch.Tensor) -> torch.Tensor:
        """Deformed mesh -> normalized lightcurves [C, P] (convex, or occluded)."""
        normals, areas = self.facet_geometry(vertices)
        forward_model = AsteroidForwardModel(
            normals=normals,
            lambert_weight=self.lambert_weight,
        )
        matrix = forward_model.build_forward_matrix(
            phases=self.phases, e_sun=self.e_sun, e_obs=self.e_obs
        )
        if self.occlusion_enabled and self.state["active"]:
            if self.state["mask"] is None or self.state["calls"] % self.occlusion_refresh_every == 0:
                self.state["mask"] = self.compute_visibility_mask(vertices)
            self.state["calls"] += 1
            matrix = matrix * self.state["mask"]
        raw = torch.einsum("cpf,f->cp", matrix, areas)
        return raw / raw.mean(dim=1, keepdim=True).clamp_min(1e-12)

    def __call__(self, vertices: torch.Tensor) -> torch.Tensor:
        return self.predicted_lightcurves(vertices)
