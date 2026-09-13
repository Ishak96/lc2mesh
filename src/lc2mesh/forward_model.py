"""Differentiable asteroid forward photometric model (Lommel-Seeliger + Lambert)."""

from __future__ import annotations

from typing import Mapping

import torch


def normalize_vectors(
    vectors: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize vectors to unit length along ``dim``."""
    return vectors / vectors.norm(dim=dim, keepdim=True).clamp_min(float(eps))


def observer_geometry_to_unit_vector(
    theta_deg: float,
    alpha_deg: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert HAC observer angles to a unit direction vector."""
    theta = torch.deg2rad(
        torch.tensor(float(theta_deg) - 180.0, dtype=dtype, device=device)
    )
    alpha = torch.deg2rad(torch.tensor(float(alpha_deg), dtype=dtype, device=device))

    vec = torch.stack(
        [
            torch.cos(theta) * torch.cos(alpha),
            torch.sin(theta) * torch.cos(alpha),
            torch.sin(alpha),
        ],
        dim=0,
    )
    return normalize_vectors(vec, dim=0)


def observers_to_unit_tensor(
    observer_map: Mapping[str, object],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[list[str], torch.Tensor]:
    """Build ordered observer names and stacked observer vectors."""
    names = list(observer_map.keys())
    e_obs = [
        observer_geometry_to_unit_vector(
            float(observer_map[name].theta),
            float(observer_map[name].alpha),
            device=device,
            dtype=dtype,
        )
        for name in names
    ]
    return names, torch.stack(e_obs, dim=0)


class AsteroidForwardModel(torch.nn.Module):
    """Differentiable LS+Lambert asteroid forward operator implemented in PyTorch."""

    def __init__(
        self,
        normals: torch.Tensor,
        *,
        lambert_weight: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        normals_t = normalize_vectors(normals.to(dtype=torch.float32), dim=1, eps=eps)
        self.register_buffer("normals", normals_t)

        self.lambert_weight = float(lambert_weight)
        self.eps = float(eps)

    @property
    def n_facets(self) -> int:
        return int(self.normals.shape[0])

    @staticmethod
    def rotate_vectors_z(vectors: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """Rotate vectors around +z for all phase angles."""
        c = torch.cos(phases)
        s = torch.sin(phases)

        if vectors.ndim == 1:
            x = c * vectors[0] - s * vectors[1]
            y = s * vectors[0] + c * vectors[1]
            z = torch.ones_like(x) * vectors[2]
            return torch.stack([x, y, z], dim=-1)

        if vectors.ndim == 2:
            x = c[:, None] * vectors[None, :, 0] - s[:, None] * vectors[None, :, 1]
            y = s[:, None] * vectors[None, :, 0] + c[:, None] * vectors[None, :, 1]
            z = vectors[None, :, 2].expand_as(x)
            return torch.stack([x, y, z], dim=-1)

        raise ValueError("vectors must have shape [3] or [N, 3]")

    def build_forward_matrix_components(
        self,
        phases: torch.Tensor,
        e_sun: torch.Tensor,
        e_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build LS/Lambert component matrices ``(A_ls, A_prod)`` of shape [C, P, F].

        The full matrix for any Lambert weight ``w`` is ``A(w) = A_ls + w * A_prod``;
        visibility masks and phase factors are computed only once, which makes
        Lambert-weight sweeps cheap.
        """
        phases = phases.to(dtype=self.normals.dtype, device=self.normals.device)
        e_sun = normalize_vectors(
            e_sun.to(dtype=self.normals.dtype, device=self.normals.device),
            dim=-1,
            eps=self.eps,
        )
        e_obs = normalize_vectors(
            e_obs.to(dtype=self.normals.dtype, device=self.normals.device),
            dim=-1,
            eps=self.eps,
        )

        e_sun_rot = self.rotate_vectors_z(e_sun, phases)  # [P, 3]
        e_obs_rot = self.rotate_vectors_z(e_obs, phases)  # [P, C, 3]

        mu0 = torch.einsum("pd,fd->pf", e_sun_rot, self.normals)[:, None, :]  # [P,1,F]
        mu = torch.einsum("pcd,fd->pcf", e_obs_rot, self.normals)  # [P,C,F]

        mu0_pos = torch.clamp(mu0, min=0.0)
        mu_pos = torch.clamp(mu, min=0.0)
        prod = mu_pos * mu0_pos

        ls_term = prod / (mu_pos + mu0_pos + self.eps)

        cos_alpha = torch.einsum("pd,pcd->pc", e_sun_rot, e_obs_rot)
        cos_alpha = torch.clamp(cos_alpha, min=-1.0, max=1.0)
        alpha = torch.acos(cos_alpha)

        visible = (mu0 > 0.0) & (mu > 0.0)
        a_ls = torch.where(
            visible,
            ls_term,
            torch.zeros_like(ls_term),
        )
        a_prod = torch.where(
            visible,
            prod,
            torch.zeros_like(prod),
        )
        return (
            a_ls.permute(1, 0, 2).contiguous(),
            a_prod.permute(1, 0, 2).contiguous(),
        )  # [C,P,F] each

    def build_forward_matrix(
        self,
        phases: torch.Tensor,
        e_sun: torch.Tensor,
        e_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Build LS+Lambert matrix ``A`` with shape [n_cams, n_phases, n_facets]."""
        a_ls, a_prod = self.build_forward_matrix_components(
            phases=phases, e_sun=e_sun, e_obs=e_obs
        )
        if self.lambert_weight == 0.0:
            return a_ls
        return a_ls + self.lambert_weight * a_prod
