from __future__ import annotations

from dataclasses import dataclass
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable, Literal, Any
from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper

import torch
import torch.nn as nn

# PDE component registry local to this file
_PDE_COMPONENT_REGISTRY: Dict[str, Callable[..., "PDEComponent"]] = {}

# Spatial backend registry local to this file
_SPATIAL_BACKEND_REGISTRY: Dict[str, Callable[..., "SpatialDerivativeBackend"]] = {}

def register_pde_component(name: str):
    def deco(fn):
        _PDE_COMPONENT_REGISTRY[name] = fn
        return fn
    return deco

def register_spatial_backend(name: str):
    def deco(fn):
        _SPATIAL_BACKEND_REGISTRY[name] = fn
        return fn
    return deco

# ----------------------------
# Differentiation backends
# ----------------------------

class SpatialDerivativeBackend(nn.Module):
    """
    Abstract spatial derivative backend.

    Given a scalar/tensor field with shape:
        (B, T, *spatial)   or   (B, *spatial)
    returns spatial derivatives with matching leading dims.

    NOTE:
      * For benchmarking, keep the API minimal and predictable.
      * You can add jacobians, vector calculus ops, curvilinear metrics later.
    """
    def grad(self, u: torch.Tensor, dx: Union[float, Tuple[float, ...]]) -> List[torch.Tensor]:
        raise NotImplementedError

    def div(self, v: List[torch.Tensor], dx: Union[float, Tuple[float, ...]]) -> torch.Tensor:
        raise NotImplementedError

    def laplacian(self, u: torch.Tensor, dx: Union[float, Tuple[float, ...]]) -> torch.Tensor:
        raise NotImplementedError

@register_spatial_backend("FDSpatialDerivatives")
class FDSpatialDerivatives(SpatialDerivativeBackend):
    """
    Finite-difference spatial derivatives on a uniform grid.

    Supports 1D/2D/3D via specifying dx as float or tuple of floats.
    Uses 2nd-order centered differences in the interior and 2nd-order
    one-sided stencils at boundaries.

    You can swap stencils/order later without changing loss code.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _as_tuple(dx: Union[float, Tuple[float, ...]], ndim: int) -> Tuple[float, ...]:
        if isinstance(dx, (float, int)):
            return (float(dx),) * ndim
        if len(dx) != ndim:
            raise ValueError(f"dx has len {len(dx)} but spatial ndim is {ndim}")
        return tuple(float(x) for x in dx)

    @staticmethod
    def _diff1(u: torch.Tensor, h: float, dim: int) -> torch.Tensor:
        # 2nd order centered interior, 2nd order one-sided boundaries
        # u shape: (..., N, ...)
        # centered
        up = u.roll(-1, dims=dim)
        um = u.roll(1, dims=dim)
        du = (up - um) / (2.0 * h)

        # fix boundaries with one-sided 2nd order: (-3u0 + 4u1 - u2)/(2h) and (3uN - 4uN-1 + uN-2)/(2h)
        slc0 = [slice(None)] * u.ndim
        slc1 = [slice(None)] * u.ndim
        slc2 = [slice(None)] * u.ndim
        slcN = [slice(None)] * u.ndim
        slcNm1 = [slice(None)] * u.ndim
        slcNm2 = [slice(None)] * u.ndim

        slc0[dim] = 0
        slc1[dim] = 1
        slc2[dim] = 2
        slcN[dim] = -1
        slcNm1[dim] = -2
        slcNm2[dim] = -3

        du0 = (-3.0 * u[tuple(slc0)] + 4.0 * u[tuple(slc1)] - 1.0 * u[tuple(slc2)]) / (2.0 * h)
        duN = ( 3.0 * u[tuple(slcN)] - 4.0 * u[tuple(slcNm1)] + 1.0 * u[tuple(slcNm2)]) / (2.0 * h)

        du = du.clone()
        du[tuple(slc0)] = du0
        du[tuple(slcN)] = duN
        return du

    @staticmethod
    def _diff2(u: torch.Tensor, h: float, dim: int) -> torch.Tensor:
        # 2nd derivative: centered interior; simple 2nd order one-sided at boundaries.
        up = u.roll(-1, dims=dim)
        um = u.roll(1, dims=dim)
        d2 = (up - 2.0 * u + um) / (h * h)

        # one-sided 2nd order at boundaries:
        # u_xx(0) ~ (2u0 - 5u1 + 4u2 - u3)/h^2
        # u_xx(N) ~ (2uN - 5uN-1 + 4uN-2 - uN-3)/h^2
        slc0 = [slice(None)] * u.ndim
        slc1 = [slice(None)] * u.ndim
        slc2 = [slice(None)] * u.ndim
        slc3 = [slice(None)] * u.ndim
        slcN = [slice(None)] * u.ndim
        slcNm1 = [slice(None)] * u.ndim
        slcNm2 = [slice(None)] * u.ndim
        slcNm3 = [slice(None)] * u.ndim

        slc0[dim] = 0
        slc1[dim] = 1
        slc2[dim] = 2
        slc3[dim] = 3
        slcN[dim] = -1
        slcNm1[dim] = -2
        slcNm2[dim] = -3
        slcNm3[dim] = -4

        d20 = (2.0 * u[tuple(slc0)] - 5.0 * u[tuple(slc1)] + 4.0 * u[tuple(slc2)] - 1.0 * u[tuple(slc3)]) / (h * h)
        d2N = (2.0 * u[tuple(slcN)] - 5.0 * u[tuple(slcNm1)] + 4.0 * u[tuple(slcNm2)] - 1.0 * u[tuple(slcNm3)]) / (h * h)

        d2 = d2.clone()
        d2[tuple(slc0)] = d20
        d2[tuple(slcN)] = d2N
        return d2

    def grad(self, u: torch.Tensor, dx: Union[float, Tuple[float, ...]]) -> List[torch.Tensor]:
        # spatial dims are the last N dims
        # u: (B,T,*spatial) or (B,*spatial)
        spatial_ndim = u.ndim - (2 if u.ndim >= 3 else 1)  # conservative heuristic
        # Better: user passes `data_dim` into LossComponent; we'll use u.ndim-2 if u has (B,T,...) shape.
        if u.ndim >= 3:
            spatial_ndim = u.ndim - 2
            first_spatial_dim = 2
        else:
            spatial_ndim = u.ndim - 1
            first_spatial_dim = 1

        dx_tup = self._as_tuple(dx, spatial_ndim)
        grads = []
        for i in range(spatial_ndim):
            dim = first_spatial_dim + i
            grads.append(self._diff1(u, dx_tup[i], dim))
        return grads

    def div(self, v: List[torch.Tensor], dx: Union[float, Tuple[float, ...]]) -> torch.Tensor:
        if len(v) == 0:
            raise ValueError("v must have at least one component")
        # infer spatial ndim from list length
        spatial_ndim = len(v)
        # infer first spatial dim from tensor rank
        u = v[0]
        first_spatial_dim = 2 if u.ndim >= 3 else 1
        dx_tup = self._as_tuple(dx, spatial_ndim)
        out = 0.0
        for i in range(spatial_ndim):
            out = out + self._diff1(v[i], dx_tup[i], first_spatial_dim + i)
        return out

    def laplacian(self, u: torch.Tensor, dx: Union[float, Tuple[float, ...]]) -> torch.Tensor:
        spatial_ndim = u.ndim - (2 if u.ndim >= 3 else 1)
        first_spatial_dim = 2 if u.ndim >= 3 else 1
        dx_tup = self._as_tuple(dx, spatial_ndim)
        out = 0.0
        for i in range(spatial_ndim):
            out = out + self._diff2(u, dx_tup[i], first_spatial_dim + i)
        return out


# Optional: coordinate-AD backend (for coordinate MLP PINNs).
# This is included for completeness; for AR rollouts you usually won't use it. 
@register_spatial_backend("CoordADSpatialDerivatives")
class CoordADSpatialDerivatives(SpatialDerivativeBackend):
    """
    Coordinate-based AD spatial derivatives for coordinate networks u(x,t).
    Expects u is produced with coords that require_grad=True.

    In an AR/grid rollout benchmark, this is typically not the hot path,
    but it's useful if you also want a "true vanilla PINN" baseline later.
    """
    def __init__(self, coords: torch.Tensor, spatial_dim: int):
        """
        coords: Tensor of shape (..., spatial_dim) with requires_grad=True
        spatial_dim: number of spatial coordinates
        """
        super().__init__()
        self.coords = coords
        self.spatial_dim = spatial_dim

    def grad(self, u: torch.Tensor, dx=None) -> List[torch.Tensor]:
        grads = []
        for i in range(self.spatial_dim):
            gi = torch.autograd.grad(
                u, self.coords, grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True, allow_unused=False
            )[0][..., i]
            grads.append(gi)
        return grads

    def div(self, v: List[torch.Tensor], dx=None) -> torch.Tensor:
        # assumes v[i] corresponds to component i
        out = 0.0
        for i, vi in enumerate(v):
            dvi = torch.autograd.grad(
                vi, self.coords, grad_outputs=torch.ones_like(vi),
                create_graph=True, retain_graph=True, allow_unused=False
            )[0][..., i]
            out = out + dvi
        return out

    def laplacian(self, u: torch.Tensor, dx=None) -> torch.Tensor:
        grads = self.grad(u)
        out = 0.0
        for i, gi in enumerate(grads):
            dgi = torch.autograd.grad(
                gi, self.coords, grad_outputs=torch.ones_like(gi),
                create_graph=True, retain_graph=True, allow_unused=False
            )[0][..., i]
            out = out + dgi
        return out


# ----------------------------
# Time derivative (FD only)
# ----------------------------

def fd_time_derivative(
    u: torch.Tensor,
    dt: float,
    scheme: Literal["forward1", "center2"] = "center2",
    eval_on: Literal["interior", "all"] = "interior",
    prev: Optional[torch.Tensor] = None,  # optional previous frame
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute u_t along time axis (dim=1) for u shaped (B, T, *spatial).
    If `prev` is given, it is prepended to form forward differences so that
    derivatives are available for every prediction frame (len = T).
    """
    if u.ndim < 3:
        raise ValueError("Expected u with shape (B, T, *spatial)")

    T = u.shape[1]

    if scheme == "forward1":
        if prev is not None:
            # prev: (B, *spatial) or (B,1,*spatial)
            if prev.ndim == u.ndim - 1:
                prev = prev.unsqueeze(1)
            elif prev.ndim != u.ndim or prev.shape[1] != 1:
                raise ValueError("prev must have shape (B, *spatial) or (B,1,*spatial)")
            prev = prev.to(u.device, u.dtype)
            u_aug = torch.cat([prev, u], dim=1)          # (B, T+1, *spatial)
            ut = (u_aug[:, 1:] - u_aug[:, :-1]) / dt      # (B, T, *spatial)
            t_idx = torch.arange(0, T, device=u.device, dtype=torch.long)
            return ut, t_idx
        # standard forward diff (length T-1)
        if T < 2:
            raise ValueError("Need T>=2 for forward1 time derivative (or provide prev)")
        ut = (u[:, 1:] - u[:, :-1]) / dt
        t_idx = torch.arange(0, T - 1, device=u.device, dtype=torch.long)
        return ut, t_idx

    # center2 unchanged...
    if T < 3 and scheme == "center2":
        raise ValueError("Need T>=3 for center2 time derivative")
    if scheme == "center2" and eval_on == "interior":
        ut = (u[:, 2:] - u[:, :-2]) / (2.0 * dt)
        t_idx = torch.arange(1, T - 1, device=u.device, dtype=torch.long)
        return ut, t_idx
    if scheme == "center2" and eval_on == "all":
        ut = torch.empty_like(u)
        ut[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dt)
        ut[:, 0] = (-3.0 * u[:, 0] + 4.0 * u[:, 1] - u[:, 2]) / (2.0 * dt)
        ut[:, -1] = (3.0 * u[:, -1] - 4.0 * u[:, -2] + u[:, -3]) / (2.0 * dt)
        t_idx = torch.arange(0, T, device=u.device, dtype=torch.long)
        return ut, t_idx

    raise ValueError(f"Unknown scheme: {scheme}")


# ----------------------------
# PDE components (modular)
# ----------------------------

@dataclass
class DifferentialOps:
    dt: float
    dx: Union[float, Tuple[float, ...]]
    spatial_backend: SpatialDerivativeBackend
    time_scheme: Literal["forward1", "center2"] = "center2"

    def time_derivative(
        self,
        u: torch.Tensor,
        eval_on: Literal["interior", "all"] = "interior",
        prev: Optional[torch.Tensor] = None,
    ):
        return fd_time_derivative(u, dt=self.dt, scheme=self.time_scheme, eval_on=eval_on, prev=prev)

    def grad(self, u: torch.Tensor) -> List[torch.Tensor]:
        return self.spatial_backend.grad(u, self.dx)

    def div(self, v: List[torch.Tensor]) -> torch.Tensor:
        return self.spatial_backend.div(v, self.dx)

    def laplacian(self, u: torch.Tensor) -> torch.Tensor:
        return self.spatial_backend.laplacian(u, self.dx)

class PDEComponent(nn.Module):
    """
    Base class for a single PDE residual component.
    """
    default_name: str = "pde_component"
    supported_spatial_dims: Tuple[int, ...] = (1, 2, 3)
    required_fields: Tuple[str, ...] = ()
    required_time_fields: Tuple[str, ...] = ()
    required_grad_fields: Tuple[str, ...] = ()
    required_laplacian_fields: Tuple[str, ...] = ()

    def __init__(self, name: Optional[str] = None, spatial_dim: int = 2, **params: Any):
        super().__init__()
        if spatial_dim not in self.supported_spatial_dims:
            raise ValueError(
                f"{self.__class__.__name__} only supports spatial_dim in {self.supported_spatial_dims}, "
                f"got {spatial_dim}"
            )
        self.spatial_dim = spatial_dim
        self.name = name or self.default_name
        self.params = params

    def residual_scale(self, refs: Dict[str, float]) -> Optional[float]:
        return None

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: "DerivativeCache",
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class DerivativeCache:
    t_idx: torch.Tensor
    fields_n: Dict[str, torch.Tensor]
    time: Dict[str, torch.Tensor]
    grads: Dict[str, List[torch.Tensor]]
    laplacian: Dict[str, torch.Tensor]


def _align_r_coord(
    r_coord: torch.Tensor,
    t_idx: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if r_coord.ndim == 4:
        if r_coord.shape[1] != target.shape[1]:
            if r_coord.shape[1] > t_idx.max().item():
                r_coord = r_coord[:, t_idx]
            elif r_coord.shape[1] == 1:
                r_coord = r_coord.expand(target.shape[0], target.shape[1], *r_coord.shape[-2:])
        if r_coord.shape[0] == 1 and target.shape[0] > 1:
            r_coord = r_coord.expand(target.shape[0], *r_coord.shape[1:])
        return r_coord
    if r_coord.ndim == 2:
        r_coord = r_coord[None, None, ...].expand(target.shape[0], target.shape[1], *r_coord.shape)
        return r_coord
    raise ValueError(f"Unsupported r_coord shape: {tuple(r_coord.shape)}")


def _axis_labels_from_dim(n_spatial: int) -> List[str]:
    """
    Return axis labels in tensor order (Z, Y, X).
    1D -> ["x"], 2D -> ["y", "x"], 3D -> ["z", "y", "x"].
    """
    if n_spatial == 1:
        return ["x"]
    if n_spatial == 2:
        return ["y", "x"]
    if n_spatial == 3:
        return ["z", "y", "x"]
    raise ValueError(f"Unsupported spatial dim: {n_spatial}")


def _boundary_axis_from_name(boundary: str, n_spatial: int) -> Tuple[int, str]:
    """
    Map boundary name to physical axis index (x=0,y=1,z=2) and side ('min'/'max').
    """
    boundary = boundary.lower()
    side_map = {
        "west": (0, "min"),
        "east": (0, "max"),
        "south": (1, "min"),
        "north": (1, "max"),
        "bottom": (2, "min"),
        "top": (2, "max"),
    }
    if boundary not in side_map:
        raise ValueError(
            f"Unknown boundary '{boundary}'. Expected one of: {list(side_map.keys())}."
        )
    axis, side = side_map[boundary]
    if axis >= n_spatial:
        raise ValueError(
            f"Boundary '{boundary}' is not valid for spatial_dim={n_spatial}."
        )
    return axis, side


def _physical_to_tensor_axis(axis: int, n_spatial: int) -> int:
    """
    Convert physical axis index (x=0,y=1,z=2) to tensor spatial index.
    Tensor spatial order is (Z, Y, X).
    """
    if n_spatial == 1:
        mapping = [0]
    elif n_spatial == 2:
        mapping = [1, 0]
    elif n_spatial == 3:
        mapping = [2, 1, 0]
    else:
        raise ValueError(f"Unsupported spatial dim: {n_spatial}")
    if axis < 0 or axis >= n_spatial:
        raise ValueError(f"Axis {axis} is invalid for spatial_dim={n_spatial}.")
    return mapping[axis]


def _boundary_mask(field: torch.Tensor, axis_index: int, side: str) -> torch.Tensor:
    mask = torch.zeros_like(field)
    slc = [slice(None)] * field.ndim
    slc[2 + axis_index] = 0 if side == "min" else -1
    mask[tuple(slc)] = 1.0
    return mask


def _normalize_field_token(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_field_name(requested: str, available: List[str]) -> Optional[str]:
    if requested in available:
        return requested

    req_norm = _normalize_field_token(requested)
    norm_to_names: Dict[str, List[str]] = {}
    for n in available:
        norm_to_names.setdefault(_normalize_field_token(n), []).append(n)

    candidates = norm_to_names.get(req_norm, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous field mapping for '{requested}'. Candidates: {candidates}. "
            "Please set explicit field names in component config."
        )
    return None


def _default_velocity_fields(spatial_dim: int) -> List[str]:
    mapping = {
        1: ["Velocity_X"],
        2: ["Velocity_X", "Velocity_Y"],
        3: ["Velocity_X", "Velocity_Y", "Velocity_Z"],
    }
    if spatial_dim not in mapping:
        raise ValueError(f"Unsupported spatial dim: {spatial_dim}")
    return mapping[spatial_dim]


class _MultiMaterialEulerBase(PDEComponent):
    supported_spatial_dims = (2, 3)

    def __init__(
        self,
        name: Optional[str] = None,
        spatial_dim: int = 2,
        density_field: str = "Density",
        pressure_field: str = "Pressure",
        velocity_fields: Optional[List[str]] = None,
        energy_field: str = "Energy",
        alpha_field: str = "Diffuse_Volume_Fraction_1",
        **params: Any,
    ):
        super().__init__(name=name, spatial_dim=spatial_dim, **params)
        self.density_field = str(density_field)
        self.pressure_field = str(pressure_field)
        self.velocity_fields = list(velocity_fields) if velocity_fields is not None else _default_velocity_fields(spatial_dim)
        if len(self.velocity_fields) != spatial_dim:
            raise ValueError(
                f"{self.__class__.__name__}: velocity_fields length {len(self.velocity_fields)} "
                f"must match spatial_dim={spatial_dim}."
            )
        self.energy_field = str(energy_field)
        self.alpha_field = str(alpha_field)

    def _dx(self, derivs: DerivativeCache, field: str, axis_phys: int) -> torch.Tensor:
        axis_tensor = _physical_to_tensor_axis(axis_phys, self.spatial_dim)
        return derivs.grads[field][axis_tensor]

    def _grad_phys(self, derivs: DerivativeCache, field: str) -> List[torch.Tensor]:
        return [self._dx(derivs, field, a) for a in range(self.spatial_dim)]


@register_pde_component("pde/mmEulerMass")
class MultiMaterialEulerMassConservative(_MultiMaterialEulerBase):
    """
    Conservative mixture-mass equation in Cartesian coordinates:
      ∂t(ρ) + ∇·(ρu) = 0
    """
    default_name = "mm_mass"

    def __init__(
        self,
        name: Optional[str] = None,
        spatial_dim: int = 2,
        density_field: str = "Density",
        velocity_fields: Optional[List[str]] = None,
        **params: Any,
    ):
        super().__init__(
            name=name or self.default_name,
            spatial_dim=spatial_dim,
            density_field=density_field,
            velocity_fields=velocity_fields,
            **params,
        )
        self.required_fields = (self.density_field, *self.velocity_fields)
        self.required_time_fields = (self.density_field,)
        self.required_grad_fields = (self.density_field, *self.velocity_fields)

    def residual_scale(self, refs: Dict[str, float]) -> Optional[float]:
        ref_rho = refs.get("density", 1.0)
        ref_u = refs.get("velocity", 1.0)
        ref_L = refs.get("length", 1.0)
        eps = 1e-12
        ref_t = ref_L / max(ref_u, eps)
        return max(ref_rho / ref_t, ref_rho * ref_u / ref_L)

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        rho = fields[self.density_field]
        rho_t = derivs.time[self.density_field]

        div_flux = 0.0
        for axis in range(self.spatial_dim):
            u_i = fields[self.velocity_fields[axis]]
            drho_dxi = self._dx(derivs, self.density_field, axis)
            dui_dxi = self._dx(derivs, self.velocity_fields[axis], axis)
            div_flux = div_flux + u_i * drho_dxi + rho * dui_dxi

        return rho_t + div_flux


@register_pde_component("pde/mmEulerMomentum")
class MultiMaterialEulerMomentumConservative(_MultiMaterialEulerBase):
    """
    Conservative momentum equation in Cartesian coordinates for component j:
      ∂t(ρu_j) + Σ_i ∂_{x_i}(ρu_i u_j + p δ_{ij}) = 0
    """
    default_name = "mm_momentum"

    def __init__(
        self,
        name: Optional[str] = None,
        spatial_dim: int = 2,
        direction: str = "x",
        density_field: str = "Density",
        pressure_field: str = "Pressure",
        velocity_fields: Optional[List[str]] = None,
        artificial_viscosity: float = 0.0,
        **params: Any,
    ):
        axis_by_dir = {"x": 0, "y": 1, "z": 2}
        if direction not in axis_by_dir:
            raise ValueError("MultiMaterialEulerMomentumConservative requires direction='x', 'y', or 'z'.")
        axis = axis_by_dir[direction]
        if axis >= spatial_dim:
            raise ValueError(
                f"direction='{direction}' is invalid for spatial_dim={spatial_dim}."
            )
        self.direction = direction
        self.direction_axis = axis
        self.artificial_viscosity = float(artificial_viscosity)
        super().__init__(
            name=name or f"{self.default_name}_{direction}",
            spatial_dim=spatial_dim,
            density_field=density_field,
            pressure_field=pressure_field,
            velocity_fields=velocity_fields,
            **params,
        )
        self.required_fields = (self.density_field, self.pressure_field, *self.velocity_fields)
        self.required_time_fields = (self.density_field, self.velocity_fields[self.direction_axis])
        self.required_grad_fields = (self.density_field, self.pressure_field, *self.velocity_fields)
        self.required_laplacian_fields = (self.velocity_fields[self.direction_axis],)

    def residual_scale(self, refs: Dict[str, float]) -> Optional[float]:
        ref_rho = refs.get("density", 1.0)
        ref_u = refs.get("velocity", 1.0)
        ref_p = refs.get("pressure", 1.0)
        ref_L = refs.get("length", 1.0)
        eps = 1e-12
        ref_t = ref_L / max(ref_u, eps)
        return max(
            ref_rho * ref_u / ref_t,
            ref_rho * ref_u * ref_u / ref_L,
            ref_p / ref_L,
        )

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        params = params or {}
        rho = fields[self.density_field]
        p = fields[self.pressure_field]
        u_j = fields[self.velocity_fields[self.direction_axis]]

        rho_t = derivs.time[self.density_field]
        u_j_t = derivs.time[self.velocity_fields[self.direction_axis]]
        mom_t = rho_t * u_j + rho * u_j_t

        div_flux = 0.0
        for axis in range(self.spatial_dim):
            u_i = fields[self.velocity_fields[axis]]
            drho_dxi = self._dx(derivs, self.density_field, axis)
            dui_dxi = self._dx(derivs, self.velocity_fields[axis], axis)
            duj_dxi = self._dx(derivs, self.velocity_fields[self.direction_axis], axis)
            dp_dxi = self._dx(derivs, self.pressure_field, axis)

            term = u_i * u_j * drho_dxi + rho * u_j * dui_dxi + rho * u_i * duj_dxi
            if axis == self.direction_axis:
                term = term + dp_dxi
            div_flux = div_flux + term

        nu = float(self.artificial_viscosity)
        if "mm_momentum_artificial_viscosity" in params:
            nu = float(params["mm_momentum_artificial_viscosity"])
        dir_key = f"mm_momentum_{self.direction}_artificial_viscosity"
        if dir_key in params:
            nu = float(params[dir_key])
        if "artificial_viscosity" in params:
            nu = float(params["artificial_viscosity"])

        if abs(nu) > 0.0:
            lap_u_j = derivs.laplacian[self.velocity_fields[self.direction_axis]]
            return mom_t + div_flux - nu * lap_u_j

        return mom_t + div_flux


@register_pde_component("pde/mmEulerMomentumX")
class MultiMaterialEulerMomentumXConservative(MultiMaterialEulerMomentumConservative):
    default_name = "mm_momentum_x"

    def __init__(self, name: Optional[str] = None, spatial_dim: int = 2, **params: Any):
        super().__init__(
            name=name or self.default_name,
            spatial_dim=spatial_dim,
            direction="x",
            **params,
        )


@register_pde_component("pde/mmEulerMomentumY")
class MultiMaterialEulerMomentumYConservative(MultiMaterialEulerMomentumConservative):
    default_name = "mm_momentum_y"

    def __init__(self, name: Optional[str] = None, spatial_dim: int = 2, **params: Any):
        super().__init__(
            name=name or self.default_name,
            spatial_dim=spatial_dim,
            direction="y",
            **params,
        )


@register_pde_component("pde/mmEulerMomentumZ")
class MultiMaterialEulerMomentumZConservative(MultiMaterialEulerMomentumConservative):
    default_name = "mm_momentum_z"

    def __init__(self, name: Optional[str] = None, spatial_dim: int = 3, **params: Any):
        super().__init__(
            name=name or self.default_name,
            spatial_dim=spatial_dim,
            direction="z",
            **params,
        )


@register_pde_component("pde/mmEulerEnergy")
class MultiMaterialEulerEnergyConservative(_MultiMaterialEulerBase):
    """
    Conservative total-energy equation in Cartesian coordinates:
      ∂t(E) + ∇·((E + p)u) = 0

    If `energy_from_eos=True`, E is reconstructed using a stiffened-gas closure.
    """
    default_name = "mm_energy"

    def __init__(
        self,
        name: Optional[str] = None,
        spatial_dim: int = 2,
        density_field: str = "Density",
        pressure_field: str = "Pressure",
        velocity_fields: Optional[List[str]] = None,
        energy_field: str = "Energy",
        alpha_field: str = "Diffuse_Volume_Fraction_1",
        energy_from_eos: bool = False,
        eos_mode: str = "single_phase",
        gamma: float = 1.4,
        p_inf: float = 0.0,
        gamma_gas: float = 1.4,
        gamma_liquid: float = 4.4,
        p_inf_gas: float = 0.0,
        p_inf_liquid: float = 6.0e8,
        **params: Any,
    ):
        super().__init__(
            name=name or self.default_name,
            spatial_dim=spatial_dim,
            density_field=density_field,
            pressure_field=pressure_field,
            velocity_fields=velocity_fields,
            energy_field=energy_field,
            alpha_field=alpha_field,
            **params,
        )
        self.energy_from_eos = bool(energy_from_eos)
        self.eos_mode = str(eos_mode)
        self.gamma = float(gamma)
        self.p_inf = float(p_inf)
        self.gamma_gas = float(gamma_gas)
        self.gamma_liquid = float(gamma_liquid)
        self.p_inf_gas = float(p_inf_gas)
        self.p_inf_liquid = float(p_inf_liquid)

        required_fields = [self.density_field, self.pressure_field, *self.velocity_fields]
        required_time_fields = []
        required_grad_fields = [self.pressure_field, *self.velocity_fields]

        if self.energy_from_eos:
            required_time_fields.extend([self.density_field, self.pressure_field, *self.velocity_fields])
            required_grad_fields.append(self.density_field)
            if self.eos_mode == "two_phase":
                required_fields.append(self.alpha_field)
                required_time_fields.append(self.alpha_field)
                required_grad_fields.append(self.alpha_field)
        else:
            required_fields.append(self.energy_field)
            required_time_fields.append(self.energy_field)
            required_grad_fields.append(self.energy_field)

        self.required_fields = tuple(dict.fromkeys(required_fields))
        self.required_time_fields = tuple(dict.fromkeys(required_time_fields))
        self.required_grad_fields = tuple(dict.fromkeys(required_grad_fields))

    def residual_scale(self, refs: Dict[str, float]) -> Optional[float]:
        ref_u = refs.get("velocity", 1.0)
        ref_p = refs.get("pressure", 1.0)
        ref_L = refs.get("length", 1.0)
        ref_rho = refs.get("density", 1.0)
        eps = 1e-12
        ref_t = ref_L / max(ref_u, eps)
        ref_E = max(ref_p, ref_rho * ref_u * ref_u)
        return max(ref_E / ref_t, ref_E * ref_u / ref_L, ref_p * ref_u / ref_L)

    def _single_phase_energy(
        self,
        rho: torch.Tensor,
        p: torch.Tensor,
        vel: List[torch.Tensor],
        rho_t: torch.Tensor,
        p_t: torch.Tensor,
        vel_t: List[torch.Tensor],
        rho_g: List[torch.Tensor],
        p_g: List[torch.Tensor],
        vel_g: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if self.gamma <= 1.0:
            raise ValueError("mmEulerEnergy: gamma must be > 1 for stiffened-gas closure.")
        inv = 1.0 / (self.gamma - 1.0)
        const = self.gamma * self.p_inf * inv

        q2 = 0.0
        q2_t = 0.0
        q2_g = [0.0 for _ in range(self.spatial_dim)]
        for i in range(self.spatial_dim):
            q2 = q2 + vel[i] * vel[i]
            q2_t = q2_t + 2.0 * vel[i] * vel_t[i]
            for a in range(self.spatial_dim):
                q2_g[a] = q2_g[a] + 2.0 * vel[i] * vel_g[i][a]

        rho_e = inv * p + const
        rho_e_t = inv * p_t
        rho_e_g = [inv * p_g[a] for a in range(self.spatial_dim)]

        E = rho_e + 0.5 * rho * q2
        E_t = rho_e_t + 0.5 * (rho_t * q2 + rho * q2_t)
        E_g = [rho_e_g[a] + 0.5 * (rho_g[a] * q2 + rho * q2_g[a]) for a in range(self.spatial_dim)]
        return E, E_t, E_g

    def _two_phase_energy(
        self,
        rho: torch.Tensor,
        p: torch.Tensor,
        alpha: torch.Tensor,
        vel: List[torch.Tensor],
        rho_t: torch.Tensor,
        p_t: torch.Tensor,
        alpha_t: torch.Tensor,
        vel_t: List[torch.Tensor],
        rho_g: List[torch.Tensor],
        p_g: List[torch.Tensor],
        alpha_g: List[torch.Tensor],
        vel_g: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if self.gamma_gas <= 1.0 or self.gamma_liquid <= 1.0:
            raise ValueError("mmEulerEnergy: gamma_gas and gamma_liquid must be > 1.")

        Ag = 1.0 / (self.gamma_gas - 1.0)
        Al = 1.0 / (self.gamma_liquid - 1.0)
        Bg = self.gamma_gas * self.p_inf_gas * Ag
        Bl = self.gamma_liquid * self.p_inf_liquid * Al

        A = alpha * Ag + (1.0 - alpha) * Al
        B = alpha * Bg + (1.0 - alpha) * Bl

        dA_dalpha = Ag - Al
        dB_dalpha = Bg - Bl

        q2 = 0.0
        q2_t = 0.0
        q2_g = [0.0 for _ in range(self.spatial_dim)]
        for i in range(self.spatial_dim):
            q2 = q2 + vel[i] * vel[i]
            q2_t = q2_t + 2.0 * vel[i] * vel_t[i]
            for a in range(self.spatial_dim):
                q2_g[a] = q2_g[a] + 2.0 * vel[i] * vel_g[i][a]

        rho_e = A * p + B
        rho_e_t = A * p_t + dA_dalpha * alpha_t * p + dB_dalpha * alpha_t
        rho_e_g = [
            A * p_g[a] + dA_dalpha * alpha_g[a] * p + dB_dalpha * alpha_g[a]
            for a in range(self.spatial_dim)
        ]

        E = rho_e + 0.5 * rho * q2
        E_t = rho_e_t + 0.5 * (rho_t * q2 + rho * q2_t)
        E_g = [rho_e_g[a] + 0.5 * (rho_g[a] * q2 + rho * q2_g[a]) for a in range(self.spatial_dim)]
        return E, E_t, E_g

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        rho = fields[self.density_field]
        p = fields[self.pressure_field]
        vel = [fields[vn] for vn in self.velocity_fields]

        p_g = self._grad_phys(derivs, self.pressure_field)
        du_diag = [self._dx(derivs, self.velocity_fields[a], a) for a in range(self.spatial_dim)]

        if self.energy_from_eos:
            rho_t = derivs.time[self.density_field]
            p_t = derivs.time[self.pressure_field]
            vel_t = [derivs.time[vn] for vn in self.velocity_fields]
            rho_g = self._grad_phys(derivs, self.density_field)
            vel_g = [self._grad_phys(derivs, vn) for vn in self.velocity_fields]

            if self.eos_mode == "single_phase":
                E, E_t, E_g = self._single_phase_energy(
                    rho=rho,
                    p=p,
                    vel=vel,
                    rho_t=rho_t,
                    p_t=p_t,
                    vel_t=vel_t,
                    rho_g=rho_g,
                    p_g=p_g,
                    vel_g=vel_g,
                )
            elif self.eos_mode == "two_phase":
                alpha = fields[self.alpha_field]
                alpha_t = derivs.time[self.alpha_field]
                alpha_g = self._grad_phys(derivs, self.alpha_field)
                E, E_t, E_g = self._two_phase_energy(
                    rho=rho,
                    p=p,
                    alpha=alpha,
                    vel=vel,
                    rho_t=rho_t,
                    p_t=p_t,
                    alpha_t=alpha_t,
                    vel_t=vel_t,
                    rho_g=rho_g,
                    p_g=p_g,
                    alpha_g=alpha_g,
                    vel_g=vel_g,
                )
            else:
                raise ValueError(
                    f"mmEulerEnergy: unknown eos_mode '{self.eos_mode}'. Expected 'single_phase' or 'two_phase'."
                )
        else:
            E = fields[self.energy_field]
            E_t = derivs.time[self.energy_field]
            E_g = self._grad_phys(derivs, self.energy_field)

        div_flux = 0.0
        for axis in range(self.spatial_dim):
            div_flux = div_flux + vel[axis] * (E_g[axis] + p_g[axis]) + (E + p) * du_diag[axis]

        return E_t + div_flux


@register_pde_component("pde/mmEulerVolumeFraction")
class MultiMaterialEulerVolumeFractionConservative(_MultiMaterialEulerBase):
    """
    Conservative gas volume-fraction transport in Cartesian coordinates:
      ∂t(α) + ∇·(αu) = 0
    """
    default_name = "mm_volume_fraction"

    def __init__(
        self,
        name: Optional[str] = None,
        spatial_dim: int = 2,
        alpha_field: str = "Diffuse_Volume_Fraction_1",
        velocity_fields: Optional[List[str]] = None,
        **params: Any,
    ):
        super().__init__(
            name=name or self.default_name,
            spatial_dim=spatial_dim,
            velocity_fields=velocity_fields,
            alpha_field=alpha_field,
            **params,
        )
        self.required_fields = (self.alpha_field, *self.velocity_fields)
        self.required_time_fields = (self.alpha_field,)
        self.required_grad_fields = (self.alpha_field, *self.velocity_fields)

    def residual_scale(self, refs: Dict[str, float]) -> Optional[float]:
        ref_u = refs.get("velocity", 1.0)
        ref_L = refs.get("length", 1.0)
        eps = 1e-12
        ref_t = ref_L / max(ref_u, eps)
        return max(1.0 / ref_t, ref_u / ref_L)

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        alpha = fields[self.alpha_field]
        alpha_t = derivs.time[self.alpha_field]

        div_flux = 0.0
        for axis in range(self.spatial_dim):
            u_i = fields[self.velocity_fields[axis]]
            dalpha_dxi = self._dx(derivs, self.alpha_field, axis)
            dui_dxi = self._dx(derivs, self.velocity_fields[axis], axis)
            div_flux = div_flux + u_i * dalpha_dxi + alpha * dui_dxi

        return alpha_t + div_flux


@register_pde_component("pde/unsteadyContinuity")
class UnsteadyContinuity2D(PDEComponent):
    """
    Unsteady continuity residual in 2D (primitive variables).

    Residual: rho_t + div(rho u) = 0
    """
    default_name = "continuity"
    supported_spatial_dims = (2,)
    required_fields = ("Density", "Velocity_X", "Velocity_Y")
    required_time_fields = ("Density",)
    required_grad_fields = ("Density", "Velocity_X", "Velocity_Y")

    def residual_scale(self, refs: Dict[str, float]) -> Optional[float]:
        ref_rho = refs.get("density", 1.0)
        ref_u = refs.get("velocity", 1.0)
        ref_L = refs.get("length", 1.0)
        eps = 1e-12
        ref_t = ref_L / max(ref_u, eps)
        return max(ref_rho / ref_t, ref_rho * ref_u / ref_L)

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        params = params or {}
        geometry = params.get("geometry", "planar")
        eps_r = float(params.get("eps_r", 1e-6))

        rho = fields["Density"]
        u = fields["Velocity_X"]
        v = fields["Velocity_Y"]

        rho_t = derivs.time["Density"]

        drho_dx, drho_dy = derivs.grads["Density"]
        du_dx, du_dy = derivs.grads["Velocity_X"]
        dv_dx, dv_dy = derivs.grads["Velocity_Y"]

        div_rho_u = u * drho_dx + rho * du_dx + v * drho_dy + rho * dv_dy

        if geometry == "axisymmetric":
            if "r_coord" not in params:
                raise ValueError(
                    "Axisymmetric geometry requires params['r_coord'] broadcastable to (B,T_eval,H,W) "
                    "or (1,1,H,W)."
                )
            r_coord = _align_r_coord(params["r_coord"], derivs.t_idx, rho)
            r = r_coord.clamp_min(eps_r)
            div_rho_u = div_rho_u + (rho * u) / r
        elif geometry != "planar":
            raise ValueError(f"Unknown geometry: {geometry} (expected 'planar' or 'axisymmetric')")

        return rho_t + div_rho_u


@register_pde_component("pde/eulerMomentum")
class EulerMomentum2D(PDEComponent):
    """
    Unsteady Euler momentum residual in 2D (primitive variables).
    """
    supported_spatial_dims = (2,)
    default_name = "momentum"
    required_fields = ("Density", "Pressure", "Velocity_X", "Velocity_Y")
    required_time_fields = ("Density", "Velocity_X", "Velocity_Y")
    required_grad_fields = ("Density", "Pressure", "Velocity_X", "Velocity_Y")

    def __init__(self, name: Optional[str] = None, spatial_dim: int = 2, direction: str = "x", **params: Any):
        if direction not in ("x", "y"):
            raise ValueError("EulerMomentum2D requires direction='x' or 'y'.")
        self.direction = direction
        super().__init__(name=name or f"momentum_{direction}", spatial_dim=spatial_dim, **params)

    def residual_scale(self, refs: Dict[str, float]) -> Optional[float]:
        ref_rho = refs.get("density", 1.0)
        ref_u = refs.get("velocity", 1.0)
        ref_p = refs.get("pressure", 1.0)
        ref_L = refs.get("length", 1.0)
        eps = 1e-12
        ref_t = ref_L / max(ref_u, eps)
        return max(
            ref_rho * ref_u / ref_t,
            ref_rho * ref_u * ref_u / ref_L,
            ref_p / ref_L,
        )

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        params = params or {}
        geometry = params.get("geometry", "planar")
        eps_r = float(params.get("eps_r", 1e-6))

        rho = fields["Density"]
        p = fields["Pressure"]
        u = fields["Velocity_X"]
        v = fields["Velocity_Y"]

        rho_t = derivs.time["Density"]
        u_t = derivs.time["Velocity_X"]
        v_t = derivs.time["Velocity_Y"]

        drho_dx, drho_dy = derivs.grads["Density"]
        du_dx, du_dy = derivs.grads["Velocity_X"]
        dv_dx, dv_dy = derivs.grads["Velocity_Y"]
        dp_dx, dp_dy = derivs.grads["Pressure"]

        if self.direction == "x":
            mom_t = rho_t * u + rho * u_t

            dFx_dx = u * u * drho_dx + 2.0 * rho * u * du_dx + dp_dx
            dFy_dy = u * v * drho_dy + rho * v * du_dy + rho * u * dv_dy

            div_flux = dFx_dx + dFy_dy

            if geometry == "axisymmetric":
                if "r_coord" not in params:
                    raise ValueError(
                        "Axisymmetric geometry requires params['r_coord'] broadcastable to (B,T_eval,H,W) "
                        "or (1,1,H,W)."
                    )
                r_coord = _align_r_coord(params["r_coord"], derivs.t_idx, rho)
                r = r_coord.clamp_min(eps_r)
                div_flux = div_flux + (rho * u * u + p) / r
            elif geometry != "planar":
                raise ValueError(f"Unknown geometry: {geometry} (expected 'planar' or 'axisymmetric')")

            return mom_t + div_flux

        mom_t = rho_t * v + rho * v_t

        dFx_dx = u * v * drho_dx + rho * v * du_dx + rho * u * dv_dx
        dFy_dy = v * v * drho_dy + 2.0 * rho * v * dv_dy + dp_dy

        div_flux = dFx_dx + dFy_dy

        if geometry == "axisymmetric":
            if "r_coord" not in params:
                raise ValueError(
                    "Axisymmetric geometry requires params['r_coord'] broadcastable to (B,T_eval,H,W) "
                    "or (1,1,H,W)."
                )
            r_coord = _align_r_coord(params["r_coord"], derivs.t_idx, rho)
            r = r_coord.clamp_min(eps_r)
            div_flux = div_flux + (rho * u * v) / r
        elif geometry != "planar":
            raise ValueError(f"Unknown geometry: {geometry} (expected 'planar' or 'axisymmetric')")

        return mom_t + div_flux


@register_pde_component("pde/eulerMomentumX")
class EulerMomentumX2D(EulerMomentum2D):
    """
    Unsteady Euler momentum residual in 2D, x-direction.
    """
    default_name = "momentum_x"

    def __init__(self, name: Optional[str] = None, spatial_dim: int = 2, **params: Any):
        super().__init__(name=name or self.default_name, spatial_dim=spatial_dim, direction="x", **params)


@register_pde_component("pde/eulerMomentumY")
class EulerMomentumY2D(EulerMomentum2D):
    """
    Unsteady Euler momentum residual in 2D, y-direction.
    """
    default_name = "momentum_y"

    def __init__(self, name: Optional[str] = None, spatial_dim: int = 2, **params: Any):
        super().__init__(name=name or self.default_name, spatial_dim=spatial_dim, direction="y", **params)


@register_pde_component("pde/vorticityConsistency")
class VorticityConsistency2D(PDEComponent):
    """
    Vorticity definition consistency residual in 2D.
    """
    default_name = "vorticity_consistency"
    supported_spatial_dims = (2,)
    required_fields = ("Velocity_X", "Velocity_Y", "Vorticity")
    required_grad_fields = ("Velocity_X", "Velocity_Y")

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        u_r = fields["Velocity_X"]
        u_z = fields["Velocity_Y"]
        omega = fields["Vorticity"]

        duz_dx = derivs.grads["Velocity_Y"][0]
        dur_dy = derivs.grads["Velocity_X"][1]

        curl_theta = duz_dx - dur_dy
        return omega - curl_theta


@register_pde_component("pde/debugSystemEq1")
class DebugSystemEq1(PDEComponent):
    """
    Debug system equation 1 (uses only Density, Velocity_X, Velocity_Y).
    Residual: rho_t + u_x + v_y
    """
    default_name = "debug_eq1"
    supported_spatial_dims = (2,)
    required_fields = ("Density", "Velocity_X", "Velocity_Y")
    required_time_fields = ("Density",)
    required_grad_fields = ("Velocity_X", "Velocity_Y")

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        rho_t = derivs.time["Density"]
        du_dx, _ = derivs.grads["Velocity_X"]
        _, dv_dy = derivs.grads["Velocity_Y"]
        return rho_t + du_dx + dv_dy


@register_pde_component("pde/debugSystemEq2")
class DebugSystemEq2(PDEComponent):
    """
    Debug system equation 2 (uses only Density, Velocity_X, Velocity_Y).
    Residual: div(rho * u) = (rho*u)_x + (rho*v)_y
    """
    default_name = "debug_eq2"
    supported_spatial_dims = (2,)
    required_fields = ("Density", "Velocity_X", "Velocity_Y")
    required_grad_fields = ("Density", "Velocity_X", "Velocity_Y")

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        rho = fields["Density"]
        u = fields["Velocity_X"]
        v = fields["Velocity_Y"]

        drho_dx, drho_dy = derivs.grads["Density"]
        du_dx, _ = derivs.grads["Velocity_X"]
        _, dv_dy = derivs.grads["Velocity_Y"]

        return u * drho_dx + rho * du_dx + v * drho_dy + rho * dv_dy


class _BoundaryConditionBase(PDEComponent):
    supported_spatial_dims = (1, 2, 3)

    def __init__(
        self,
        name: Optional[str] = None,
        spatial_dim: int = 2,
        fields: Optional[Union[str, List[str]]] = None,
        field_names: Optional[List[str]] = None,
        **params: Any,
    ):
        super().__init__(name=name, spatial_dim=spatial_dim, **params)
        if fields is None:
            if field_names is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires 'fields' or 'field_names' to be provided."
                )
            field_list = list(field_names)
        elif isinstance(fields, str):
            field_list = [fields]
        else:
            field_list = list(fields)

        if len(field_list) == 0:
            raise ValueError(f"{self.__class__.__name__} requires at least one field.")

        self.fields = field_list
        self.required_fields = tuple(field_list)

    def _value_for_field(
        self,
        field_name: str,
        value: Union[float, int, List[float], Tuple[float, ...], Dict[str, float]],
    ) -> float:
        if isinstance(value, dict):
            if field_name not in value:
                raise ValueError(
                    f"{self.__class__.__name__}: missing value for field '{field_name}' in value dict."
                )
            return float(value[field_name])
        if isinstance(value, (list, tuple)):
            idx = self.fields.index(field_name)
            if idx >= len(value):
                raise ValueError(
                    f"{self.__class__.__name__}: value list length {len(value)} does not cover '{field_name}'."
                )
            return float(value[idx])
        return float(value)

    @staticmethod
    def _apply_mask(residual: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        scale = mask.mean()
        return residual * mask / (scale + eps)


@register_pde_component("bc/dirichlet")
class DirichletBC(_BoundaryConditionBase):
    """
    Dirichlet boundary condition: u = value on the specified boundary.
    """
    default_name = "dirichlet"

    def __init__(
        self,
        name: Optional[str] = None,
        spatial_dim: int = 2,
        boundary: Optional[str] = None,
        value: Union[float, int, List[float], Tuple[float, ...], Dict[str, float]] = 0.0,
        fields: Optional[Union[str, List[str]]] = None,
        field_names: Optional[List[str]] = None,
        **params: Any,
    ):
        if boundary is None:
            raise ValueError("DirichletBC requires 'boundary'.")
        self.boundary = boundary
        self.value = value
        super().__init__(
            name=name or self.default_name,
            spatial_dim=spatial_dim,
            fields=fields,
            field_names=field_names,
            **params,
        )

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        params = params or {}
        label_fields = params.get("label_fields")
        if label_fields is None:
            raise ValueError(
                "DirichletBC requires label_fields to use boundary values from labels."
            )
        axis, side = _boundary_axis_from_name(self.boundary, self.spatial_dim)
        axis_index = _physical_to_tensor_axis(axis, self.spatial_dim)

        residuals = []
        for field in self.fields:
            u = fields[field]
            if field not in label_fields:
                raise ValueError(
                    f"DirichletBC: label_fields missing field '{field}'."
                )
            target = label_fields[field]
            residuals.append(u - target)

        res = torch.stack(residuals, dim=0).mean(dim=0)
        mask = _boundary_mask(res, axis_index, side)
        return self._apply_mask(res, mask)


@register_pde_component("bc/neumann")
class NeumannBC(_BoundaryConditionBase):
    """
    Neumann boundary condition: du/dn (or du/daxis) = value on boundary.
    """
    default_name = "neumann"

    def __init__(
        self,
        name: Optional[str] = None,
        spatial_dim: int = 2,
        boundary: Optional[str] = None,
        axis: Optional[int] = None,
        value: Union[float, int, List[float], Tuple[float, ...], Dict[str, float]] = 0.0,
        fields: Optional[Union[str, List[str]]] = None,
        field_names: Optional[List[str]] = None,
        **params: Any,
    ):
        if boundary is None and axis is None:
            raise ValueError("NeumannBC requires either 'boundary' or 'axis'.")
        self.boundary = boundary
        self.axis = axis
        self.value = value
        super().__init__(
            name=name or self.default_name,
            spatial_dim=spatial_dim,
            fields=fields,
            field_names=field_names,
            **params,
        )
        self.required_grad_fields = tuple(self.fields)

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if self.boundary is not None:
            axis_grad, side = _boundary_axis_from_name(self.boundary, self.spatial_dim)
            axis_mask = axis_grad
        else:
            axis_grad = int(self.axis)
            axis_mask = axis_grad
            side = None

        axis_index_grad = _physical_to_tensor_axis(axis_grad, self.spatial_dim)
        axis_index_mask = _physical_to_tensor_axis(axis_mask, self.spatial_dim)

        residuals = []
        for field in self.fields:
            grad = derivs.grads[field][axis_index_grad]
            target = self._value_for_field(field, self.value)
            residuals.append(grad - target)

        res = torch.stack(residuals, dim=0).mean(dim=0)

        if self.boundary is not None:
            if side is None:
                _, side = _boundary_axis_from_name(self.boundary, self.spatial_dim)
            mask = _boundary_mask(res, axis_index_mask, side)
        else:
            mask = _boundary_mask(res, axis_index_mask, "min") + _boundary_mask(res, axis_index_mask, "max")
        return self._apply_mask(res, mask)


@register_pde_component("bc/periodic")
class PeriodicBC(_BoundaryConditionBase):
    """
    Periodic boundary condition: u(min) = u(max) along a given axis.
    """
    default_name = "periodic"

    def __init__(
        self,
        name: Optional[str] = None,
        spatial_dim: int = 2,
        axis: Optional[int] = None,
        fields: Optional[Union[str, List[str]]] = None,
        field_names: Optional[List[str]] = None,
        **params: Any,
    ):
        if axis is None:
            raise ValueError("PeriodicBC requires 'axis'.")
        self.axis = int(axis)
        super().__init__(
            name=name or self.default_name,
            spatial_dim=spatial_dim,
            fields=fields,
            field_names=field_names,
            **params,
        )

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        derivs: DerivativeCache,
        params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        axis_index = _physical_to_tensor_axis(self.axis, self.spatial_dim)

        residuals = []
        for field in self.fields:
            u = fields[field]
            slc_min = [slice(None)] * u.ndim
            slc_max = [slice(None)] * u.ndim
            slc_min[2 + axis_index] = 0
            slc_max[2 + axis_index] = -1
            u_min = u[tuple(slc_min)]
            u_max = u[tuple(slc_max)]
            diff = u_min - u_max

            res = torch.zeros_like(u)
            res[tuple(slc_min)] = diff
            res[tuple(slc_max)] = -diff
            residuals.append(res)

        res = torch.stack(residuals, dim=0).mean(dim=0)
        mask = _boundary_mask(res, axis_index, "min") + _boundary_mask(res, axis_index, "max")
        return self._apply_mask(res, mask)

# ----------------------------
# Robust penalty (optional)
# ----------------------------

def robust_penalty(x: torch.Tensor, kind: Literal["l2", "huber"] = "l2", huber_delta: float = 1.0) -> torch.Tensor:
    """
    Elementwise penalty.
      * l2: x^2
      * huber: piecewise quadratic/linear
    """
    if kind == "l2":
        return x * x
    if kind == "huber":
        ax = x.abs()
        quad = 0.5 * (ax ** 2)
        lin = huber_delta * (ax - 0.5 * huber_delta)
        return torch.where(ax <= huber_delta, quad, lin)
    raise ValueError(f"Unknown penalty kind: {kind}")


class ResidualMasker(nn.Module):
    """Base class for residual masking strategies."""
    def __init__(self):
        super().__init__()
        self._debug_tensors: Dict[str, torch.Tensor] = {}

    def required_grad_fields(self) -> List[str]:
        return []

    def apply_residuals(
        self,
        residuals: Dict[str, torch.Tensor],
        components: List["PDEComponent"],
        derivs: "DerivativeCache",
    ) -> Dict[str, torch.Tensor]:
        return residuals

    def apply_penalty(
        self,
        penalty: torch.Tensor,
        eq_names: List[str],
        components: List["PDEComponent"],
        derivs: "DerivativeCache",
    ) -> torch.Tensor:
        return penalty

    def get_debug_tensors(self) -> Dict[str, torch.Tensor]:
        return dict(self._debug_tensors)


class EquationWeight(ResidualMasker):
    """Applies equation-weight masking based on velocity divergence."""
    def __init__(self, k1: float = 0.2):
        super().__init__()
        self.k1 = float(k1)

    def required_grad_fields(self) -> List[str]:
        return ["Velocity_X", "Velocity_Y"]

    def apply_residuals(
        self,
        residuals: Dict[str, torch.Tensor],
        components: List["PDEComponent"],
        derivs: "DerivativeCache",
    ) -> Dict[str, torch.Tensor]:
        if "Velocity_X" not in derivs.grads or "Velocity_Y" not in derivs.grads:
            raise ValueError(
                "PDEResidualLoss: equation_weight masking requires Velocity_X and Velocity_Y gradients."
            )
        du_dx, _ = derivs.grads["Velocity_X"]
        _, dv_dy = derivs.grads["Velocity_Y"]
        div_u = du_dx + dv_dy

        denom = self.k1 * (div_u.abs() - div_u) + 1.0
        eq_weight = 1.0 / (denom + 1e-12)
        self._debug_tensors = {
            "mask/equation_weight/div_u": div_u.detach(),
            "mask/equation_weight/factor": eq_weight.detach(),
        }
        masked = dict(residuals)
        for comp in components:
            if isinstance(comp, _BoundaryConditionBase):
                continue
            masked[comp.name] = masked[comp.name] * eq_weight
        return masked


class GradientAnnihilated(ResidualMasker):
    """Applies Lambda1 weighting based on |∂x U_i| for each variable."""
    def __init__(
        self,
        fields: List[str],
        alpha: Union[List[float], Tuple[float, ...], Dict[str, float], float],
        beta: Union[List[float], Tuple[float, ...], Dict[str, float], float],
        eps: float = 1e-12,
    ):
        super().__init__()
        self.fields = list(fields)
        self.alpha = self._resolve_param(alpha, self.fields, "alpha")
        self.beta = self._resolve_param(beta, self.fields, "beta")
        self.eps = float(eps)

    @staticmethod
    def _resolve_param(
        value: Union[List[float], Tuple[float, ...], Dict[str, float], float],
        fields: List[str],
        name: str,
    ) -> List[float]:
        if isinstance(value, dict):
            missing = [f for f in fields if f not in value]
            if missing:
                raise ValueError(f"Lambda1GradMasker: {name} missing for fields {missing}.")
            return [float(value[f]) for f in fields]
        if isinstance(value, (list, tuple)):
            if len(value) != len(fields):
                raise ValueError(
                    f"Lambda1GradMasker: {name} length {len(value)} does not match fields {len(fields)}."
                )
            return [float(v) for v in value]
        return [float(value)] * len(fields)

    def required_grad_fields(self) -> List[str]:
        return list(self.fields)

    def apply_penalty(
        self,
        penalty: torch.Tensor,
        eq_names: List[str],
        components: List["PDEComponent"],
        derivs: "DerivativeCache",
    ) -> torch.Tensor:
        weights = []
        debug_tensors: Dict[str, torch.Tensor] = {}
        for field_name, alpha_i, beta_i in zip(self.fields, self.alpha, self.beta):
            if field_name not in derivs.grads:
                raise ValueError(
                    f"PDEResidualLoss: lambda1_grad masking requires gradients for '{field_name}'."
                )
            grad_components = derivs.grads[field_name]
            grad_stack = torch.stack(grad_components, dim=0)
            g = torch.sqrt((grad_stack ** 2).sum(dim=0) + self.eps)
            w_i = 1.0 / (1.0 + alpha_i * (g ** beta_i))
            weights.append(w_i)
            debug_tensors[f"mask/gradient_annihilated/lambda_{field_name}"] = w_i.detach()

        lam = torch.stack(weights, dim=0).mean(dim=0)
        lam = lam.clamp_min(self.eps)
        debug_tensors["mask/gradient_annihilated/lambda_mean"] = lam.detach()

        factors = []
        for comp in components:
            factors.append(torch.ones_like(lam) if isinstance(comp, _BoundaryConditionBase) else lam)
        factor_t = torch.stack(factors, dim=2)
        for comp, factor in zip(components, factors):
            debug_tensors[f"mask/gradient_annihilated/component_factor/{comp.name}"] = factor.detach()
        self._debug_tensors = debug_tensors
        return penalty * factor_t


_RESIDUAL_MASK_REGISTRY: Dict[str, Callable[..., ResidualMasker]] = {
    "EquationWeight": EquationWeight,
    "GradientAnnihilated": GradientAnnihilated,
}


# ----------------------------
# The LossComponent: PINN residual metric for AR rollouts
# ----------------------------

class PDEResidualLoss(LossComponent):
    """
    Strong-form residual loss for grid/rollout predictions.
    """
    def __init__(
        self,
        *,
        components: Optional[List[Dict[str, Any]]] = None,
        pde: Optional[Union[str, Dict[str, Union[str, float, int]]]] = None,
        dt: float,
        dx: Union[float, Tuple[float, ...]],
        spatial_backend: Union[str, SpatialDerivativeBackend, Dict[str, Any]],
        time_scheme: Literal["forward1", "center2"] = "forward1",
        eval_time: Literal["interior", "all"] = "interior",
        # residual penalty
        penalty: Literal["l2", "huber"] = "l2",
        huber_delta: float = 1.0,
        # normalization of residual magnitudes
        residual_normalization: Literal['none', 'range', 'variance', 'std'] = 'none',
        # optional residual masking
        residual_mask: Optional[Union[
            Literal["equation_weight", "lambda1_grad"],
            Dict[str, Any],
        ]] = None,
        equation_weight_k1: float = 0.2,
        lambda1_alpha: Optional[Union[List[float], Tuple[float, ...], Dict[str, float], float]] = None,
        lambda1_beta: Optional[Union[List[float], Tuple[float, ...], Dict[str, float], float]] = None,
        lambda1_fields: Optional[List[str]] = None,
        # framework bits
        norm_helper: Optional["NormalizationHelper"] = None,
        weight: Union[float, "WeightSchedule"] = 1.0,
        name: Optional[str] = None,
        data_dim: Optional[int] = None,
        field_names: Optional[List[str]] = None,
        pde_params: Optional[Dict[str, Any]] = None,
        reference_quantities: Optional[Dict[str, float]] = None,
        residual_scale_eps: float = 1e-8,
    ):
        super().__init__(
            norm_helper=norm_helper,
            weight=weight,
            name=name or "PDEResidualLoss",
            data_dim=data_dim,
            field_names=field_names,
        )
        if field_names is None:
            raise ValueError("PDEResidualLoss requires field_names to map channels to PDE fields.")

        self.components, component_names, component_weights = self._build_components(
            components=components,
            pde=pde,
            data_dim=data_dim,
            field_names=field_names,
        )
        self.component_names = component_names
        self.ops = DifferentialOps(
            dt=float(dt),
            dx=dx,
            spatial_backend=self._build_spatial_backend(spatial_backend, data_dim),
            time_scheme=time_scheme,
        )
        self.eval_time = eval_time

        self.penalty = penalty
        self.huber_delta = float(huber_delta)

        self.residual_normalization = residual_normalization
        self.residual_mask = residual_mask
        self.equation_weight_k1 = float(equation_weight_k1)
        self.pde_params = pde_params or {}
        self._debug_mm_euler_on_labels = bool(self.pde_params.get("debug_mm_euler_on_labels", False))
        self._debug_mm_euler_max_calls = int(self.pde_params.get("debug_mm_euler_max_calls", 1))
        self._debug_mm_euler_call_count = 0
        self._debug_mm_euler_output_dir = str(
            self.pde_params.get("debug_mm_euler_output_dir", "./temporary/mm_euler_label_debug")
        )
        self._debug_mm_euler_sample_index = int(self.pde_params.get("debug_mm_euler_sample_index", 0))
        self._debug_mm_euler_time_index = int(self.pde_params.get("debug_mm_euler_time_index", 0))
        self._debug_mm_euler_max_plots = int(self.pde_params.get("debug_mm_euler_max_plots", 80))
        self._debug_mm_euler_clip_percent = float(
            self.pde_params.get("debug_mm_euler_clip_percent", 0.0)
        )
        self._debug_residual_mask_on = bool(self.pde_params.get("debug_residual_mask_on", False))
        self._debug_residual_mask_max_calls = int(self.pde_params.get("debug_residual_mask_max_calls", 1))
        self._debug_residual_mask_call_count = 0
        self._debug_residual_mask_output_dir = str(
            self.pde_params.get("debug_residual_mask_output_dir", "./temporary/residual_mask_debug")
        )
        self._debug_residual_mask_sample_index = int(self.pde_params.get("debug_residual_mask_sample_index", 0))
        self._debug_residual_mask_time_index = int(self.pde_params.get("debug_residual_mask_time_index", 0))
        self._debug_residual_mask_max_plots = int(self.pde_params.get("debug_residual_mask_max_plots", 40))
        self._debug_residual_mask_clip_percent = float(
            self.pde_params.get("debug_residual_mask_clip_percent", 0.0)
        )

        self.residual_masker: Optional[ResidualMasker] = None
        self.residual_mask_grad_fields: List[str] = []

        mask_type = self.residual_mask.get("type")
        if mask_type is not None:
            mask_cfg = {k: v for k, v in self.residual_mask.items() if k != "type"}

            if mask_type not in _RESIDUAL_MASK_REGISTRY:
                raise ValueError(
                    f"Unknown residual_mask type: {mask_type}. "
                    f"Available: {list(_RESIDUAL_MASK_REGISTRY.keys())}"
                )

            if mask_type == "EquationWeight":
                k1 = float(mask_cfg.get("equation_weight_k1", self.equation_weight_k1))
                self.residual_masker = _RESIDUAL_MASK_REGISTRY[mask_type](k1=k1)
            elif mask_type == "GradientAnnihilated":
                alpha_cfg = mask_cfg.get("lambda1_alpha", lambda1_alpha)
                beta_cfg = mask_cfg.get("lambda1_beta", lambda1_beta)
                fields_cfg = mask_cfg.get("lambda1_fields", None)
                fields_alias_cfg = mask_cfg.get("fields", None)
                if fields_cfg is not None and fields_alias_cfg is not None and list(fields_cfg) != list(fields_alias_cfg):
                    raise ValueError(
                        "PDEResidualLoss: residual_mask provides both 'lambda1_fields' and 'fields' "
                        "with different values. Please keep only one."
                    )
                if fields_cfg is None:
                    fields_cfg = fields_alias_cfg
                if fields_cfg is None:
                    fields_cfg = lambda1_fields

                if alpha_cfg is None or beta_cfg is None:
                    raise ValueError(
                        "PDEResidualLoss: GradientAnnihilated masking requires lambda1_alpha and lambda1_beta."
                    )
                fields_raw = list(fields_cfg) if fields_cfg is not None else list(self.field_names)
                if len(fields_raw) == 0:
                    raise ValueError(
                        "PDEResidualLoss: GradientAnnihilated masking requires at least one field."
                    )

                fields: List[str] = []
                unresolved_fields: List[str] = []
                for field in fields_raw:
                    resolved = _resolve_field_name(str(field), list(self.field_names))
                    if resolved is None:
                        unresolved_fields.append(str(field))
                    elif resolved not in fields:
                        fields.append(resolved)

                if unresolved_fields:
                    raise ValueError(
                        "PDEResidualLoss: GradientAnnihilated masking has unknown fields "
                        f"{unresolved_fields}. Available fields: {self.field_names}"
                    )

                self.residual_masker = _RESIDUAL_MASK_REGISTRY[mask_type](
                    fields=fields,
                    alpha=alpha_cfg,
                    beta=beta_cfg,
                )

            if self.residual_masker is not None:
                self.residual_mask_grad_fields = self.residual_masker.required_grad_fields()

        self.reference_quantities = reference_quantities or {}
        self.residual_scale_eps = float(residual_scale_eps)
        self.residual_scales = self._build_residual_scales(self.components, self.reference_quantities)

        if isinstance(weight, (int, float)):
            weight = WeightSchedule(
                base_weight=float(weight),
                component_weights=component_weights,
            )
        elif isinstance(weight, WeightSchedule) and component_weights:
            weight.component_weights = {**weight.component_weights, **component_weights}

        self.weight_schedule = weight

        component_weight_keys = list(self.weight_schedule.component_weights.keys())
        if component_weight_keys:
            missing = [k for k in component_weight_keys if k not in self.component_names]
            extra = [k for k in self.component_names if k not in self.weight_schedule.component_weights]
            if missing:
                raise ValueError(
                    "PDEResidualLoss: component weights must match configured components. "
                    f"Missing: {missing}, Extra: {extra}"
                )

    @staticmethod
    def _build_residual_scales(
        components: List[PDEComponent],
        refs: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Build characteristic scales for each PDE residual to make them O(1).
        Uses each component's `residual_scale()` if provided.
        """
        scales: Dict[str, float] = {}
        for comp in components:
            scale = comp.residual_scale(refs)
            if scale is not None:
                scales[comp.name] = float(scale)
        return scales

    @staticmethod
    def _build_components(
        components: Optional[List[Dict[str, Any]]],
        pde: Optional[Union[str, Dict[str, Union[str, float, int]]]],
        data_dim: Optional[int],
        field_names: Optional[List[str]],
    ) -> Tuple[List[PDEComponent], List[str], Dict[str, float]]:
        if components is None:
            components = []
            if pde is None:
                raise ValueError("PDEResidualLoss requires 'components' configuration.")
            if isinstance(pde, str) and pde == "EulerPrimitives2D":
                components = [
                    {"type": "pde/unsteadyContinuity"},
                    {"type": "pde/eulerMomentumX"},
                    {"type": "pde/eulerMomentumY"},
                ]
            elif isinstance(pde, str) and pde in (
                "MultiMaterialEulerConservative2D",
                "CompressibleEulerMultimaterial2D",
            ):
                components = [
                    {"type": "pde/mmEulerMass"},
                    {"type": "pde/mmEulerMomentumX"},
                    {"type": "pde/mmEulerMomentumY"},
                    {"type": "pde/mmEulerEnergy"},
                    {"type": "pde/mmEulerVolumeFraction"},
                ]
            elif isinstance(pde, str) and pde in (
                "MultiMaterialEulerConservative3D",
                "CompressibleEulerMultimaterial3D",
            ):
                components = [
                    {"type": "pde/mmEulerMass"},
                    {"type": "pde/mmEulerMomentumX"},
                    {"type": "pde/mmEulerMomentumY"},
                    {"type": "pde/mmEulerMomentumZ"},
                    {"type": "pde/mmEulerEnergy"},
                    {"type": "pde/mmEulerVolumeFraction"},
                ]
            elif isinstance(pde, str) and pde in ("debugsystem", "DebugSystem2D"):
                components = [
                    {"type": "pde/debugSystemEq1"},
                    {"type": "pde/debugSystemEq2"},
                ]
            else:
                raise ValueError(
                    "PDEResidualLoss legacy 'pde' config is unsupported. "
                    "Provide a 'components' list instead."
                )

        if not isinstance(components, list) or len(components) == 0:
            raise ValueError("PDEResidualLoss: 'components' must be a non-empty list.")

        component_instances: List[PDEComponent] = []
        component_names: List[str] = []
        component_weights: Dict[str, float] = {}
        component_name_set: set = set()

        for comp in components:
            if "type" not in comp:
                raise ValueError("PDEResidualLoss: each component must have a 'type'.")
            comp_type = str(comp["type"])
            if comp_type not in _PDE_COMPONENT_REGISTRY:
                raise ValueError(
                    f"Unknown PDE component '{comp_type}'. Available: {list(_PDE_COMPONENT_REGISTRY.keys())}"
                )
            cfg = {k: v for k, v in comp.items() if k not in ("type", "weight")}
            if "field" in cfg and "fields" not in cfg:
                cfg["fields"] = cfg.pop("field")
            if comp_type.startswith("bc/"):
                if "fields" not in cfg and "field_names" not in cfg:
                    if field_names is None:
                        raise ValueError(
                            "PDEResidualLoss: boundary condition components require field_names."
                        )
                    cfg["field_names"] = field_names
            if comp_type == "pde/eulerMomentum" and "direction" in cfg:
                raise ValueError(
                    "PDEResidualLoss: 'pde/eulerMomentum' no longer accepts a 'direction' field. "
                    "Use 'pde/eulerMomentumX' or 'pde/eulerMomentumY' instead."
                )
            name = cfg.pop("name", None)
            if "spatial_dim" not in cfg and data_dim is not None:
                cfg = {**cfg, "spatial_dim": data_dim}
            comp_instance = _PDE_COMPONENT_REGISTRY[comp_type](name=name, **cfg)
            if comp_instance.name in component_name_set:
                raise ValueError(
                    "PDEResidualLoss: duplicate component name '"
                    f"{comp_instance.name}'. Use a unique 'name' for each component."
                )
            component_name_set.add(comp_instance.name)
            component_instances.append(comp_instance)
            component_names.append(comp_instance.name)
            component_weights[comp_instance.name] = float(comp.get("weight", 1.0))

        return component_instances, component_names, component_weights

    def _tensor_to_fields(self, pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pred.shape[2] != len(self.field_names):
            raise ValueError(
                f"pred has C={pred.shape[2]} but field_names has {len(self.field_names)} entries."
            )
        return {name: pred[:, :, i, ...] for i, name in enumerate(self.field_names)}

    @staticmethod
    def _build_spatial_backend(
        backend_spec: Union[str, SpatialDerivativeBackend, Dict[str, Any]],
        data_dim: Optional[int],
    ) -> SpatialDerivativeBackend:
        if isinstance(backend_spec, SpatialDerivativeBackend):
            return backend_spec
        if isinstance(backend_spec, str):
            name = backend_spec
            cfg: Dict[str, Any] = {}
        elif isinstance(backend_spec, dict):
            name = backend_spec.get("type", None)
            if name is None:
                raise ValueError("Spatial backend dict must contain a 'type' field.")
            cfg = {k: v for k, v in backend_spec.items() if k != "type"}
        else:
            raise ValueError(f"Unsupported spatial_backend spec type: {type(backend_spec)}")

        if name not in _SPATIAL_BACKEND_REGISTRY:
            raise ValueError(f"Unknown spatial backend '{name}'. Available: {list(_SPATIAL_BACKEND_REGISTRY.keys())}")

        # FDSpatialDerivatives takes no args; CoordADSpatialDerivatives requires coords
        if name == "FDSpatialDerivatives":
            return _SPATIAL_BACKEND_REGISTRY[name](**cfg)
        if name == "CoordADSpatialDerivatives":
            if "coords" not in cfg or "spatial_dim" not in cfg:
                raise ValueError("CoordADSpatialDerivatives requires 'coords' and 'spatial_dim' in config.")
            return _SPATIAL_BACKEND_REGISTRY[name](**cfg)

        # Fallback
        return _SPATIAL_BACKEND_REGISTRY[name](**cfg)

    def _compute_component_residuals(
        self,
        fields_full: Dict[str, torch.Tensor],
        prev_fields: Optional[Dict[str, torch.Tensor]],
        label_fields_full: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[DerivativeCache, Dict[str, torch.Tensor]]:
        pde_params = {**self.pde_params, "eval_time": self.eval_time}
        derivs = self._compute_derivatives(fields_full, prev_fields)

        label_fields_n: Optional[Dict[str, torch.Tensor]] = None
        if label_fields_full is not None:
            label_fields_n = {
                name: label_fields_full[name][:, derivs.t_idx]
                for name in self.field_names
                if name in label_fields_full
            }

        residuals: Dict[str, torch.Tensor] = {}
        for comp in self.components:
            comp_params = {**comp.params, **pde_params}
            if label_fields_n is not None:
                comp_params["label_fields"] = label_fields_n
            res = comp.residual(derivs.fields_n, derivs, params=comp_params)
            residuals[comp.name] = res

        if self.residual_masker is not None:
            residuals = self.residual_masker.apply_residuals(residuals, self.components, derivs)

        return derivs, residuals

    @staticmethod
    def _is_mm_euler_component(comp: PDEComponent) -> bool:
        return isinstance(comp, _MultiMaterialEulerBase)

    def _collect_mm_debug_tensors(
        self,
        derivs: DerivativeCache,
        residuals: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        n_spatial = self.data_dim
        if n_spatial is None:
            mm_components = [c for c in self.components if self._is_mm_euler_component(c)]
            if len(mm_components) > 0:
                n_spatial = mm_components[0].spatial_dim
        if n_spatial is None:
            n_spatial = 2
        axis_labels = _axis_labels_from_dim(int(n_spatial))

        for comp in self.components:
            if not self._is_mm_euler_component(comp):
                continue
            if comp.name in residuals:
                tensors[f"residual/{comp.name}"] = residuals[comp.name]

            for field in comp.required_fields:
                if field in derivs.fields_n:
                    tensors[f"field/{comp.name}/{field}"] = derivs.fields_n[field]

            for field in comp.required_time_fields:
                if field in derivs.time:
                    tensors[f"time/{comp.name}/dt_{field}"] = derivs.time[field]

            for field in comp.required_grad_fields:
                if field in derivs.grads:
                    for i, gi in enumerate(derivs.grads[field]):
                        axis_name = axis_labels[i] if i < len(axis_labels) else f"axis{i}"
                        tensors[f"grad/{comp.name}/d{field}_d{axis_name}"] = gi

        return tensors

    @staticmethod
    def _safe_plot_name(key: str) -> str:
        out = key.replace("/", "__")
        out = out.replace(" ", "_")
        return out

    def _collect_residual_mask_debug_tensors(self) -> Dict[str, torch.Tensor]:
        if self.residual_masker is None:
            return {}
        tensors: Dict[str, torch.Tensor] = {}
        for key, val in self.residual_masker.get_debug_tensors().items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
        return tensors

    def _plot_debug_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        output_dir: str,
        sample_index: int,
        time_index: int,
        max_plots: int,
        clip_percent: float,
        run_index: int,
        tag: str,
    ) -> None:
        if not tensors:
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import TwoSlopeNorm
        except Exception:
            warnings.warn(
                "PDEResidualLoss debug plotting skipped because matplotlib is unavailable.",
                RuntimeWarning,
            )
            return

        os.makedirs(output_dir, exist_ok=True)

        max_plots = max(1, max_plots)
        n_plots = 0
        for key in sorted(tensors.keys()):
            if n_plots >= max_plots:
                break
            ten = tensors[key]
            if not isinstance(ten, torch.Tensor) or ten.ndim < 3:
                continue

            b = min(max(0, sample_index), ten.shape[0] - 1)
            t = min(max(0, time_index), ten.shape[1] - 1)
            snap = ten[b, t].detach().float().cpu()

            is_grad = key.startswith("grad/") or key.startswith("grad_")
            is_residual = key.startswith("residual/") or key.startswith("residual_")
            is_time = key.startswith("time/") or key.startswith("time_")
            is_mask = key.startswith("mask/") or key.startswith("mask_")
            needs_centered_cmap = is_grad or is_residual or is_time or is_mask

            if is_residual:
                snap = snap.abs()

            norm = None
            if needs_centered_cmap:
                max_abs_tensor = snap.abs().reshape(-1)
                clip_pct = max(0.0, min(49.9, clip_percent))
                if clip_pct > 0.0 and max_abs_tensor.numel() > 1:
                    q_hi = 1.0 - (clip_pct / 100.0)
                    max_abs = float(torch.quantile(max_abs_tensor, q_hi).item())
                else:
                    max_abs = float(max_abs_tensor.max().item())
                if max_abs < 1e-12:
                    max_abs = 1e-12
                norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

            fig = plt.figure(figsize=(5.5, 4.5))
            if snap.ndim == 1:
                ax = fig.add_subplot(111)
                ax.plot(snap.numpy())
                ax.set_xlabel("index")
                ax.set_ylabel(key)
            elif snap.ndim == 2:
                ax = fig.add_subplot(111)
                im = ax.imshow(
                    snap.numpy(),
                    origin="lower",
                    aspect="auto",
                    cmap="coolwarm",
                    norm=norm,
                )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            elif snap.ndim == 3:
                z = snap.shape[0] // 2
                ax = fig.add_subplot(111)
                im = ax.imshow(
                    snap[z].numpy(),
                    origin="lower",
                    aspect="auto",
                    cmap="coolwarm",
                    norm=norm,
                )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{key} (z={z})")
            else:
                plt.close(fig)
                continue

            if snap.ndim != 3:
                ax.set_title(key)
            fname = self._safe_plot_name(key) + ".png"
            out_path = os.path.join(output_dir, fname)
            fig.tight_layout()
            fig.savefig(out_path, dpi=140)
            plt.close(fig)
            n_plots += 1

        meta_path = os.path.join(output_dir, "debug_info.txt")
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(
                f"run={run_index} "
                f"sample={sample_index} "
                f"time={time_index} "
                f"saved_plots={n_plots} "
                f"tag={tag}\n"
            )

        print(
            f"[PDEResidualLoss] Saved {tag} debug plots to: {output_dir} "
            f"({n_plots} files)"
        )

    def _plot_mm_debug_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        self._plot_debug_tensors(
            tensors=tensors,
            output_dir=self._debug_mm_euler_output_dir,
            sample_index=self._debug_mm_euler_sample_index,
            time_index=self._debug_mm_euler_time_index,
            max_plots=self._debug_mm_euler_max_plots,
            clip_percent=self._debug_mm_euler_clip_percent,
            run_index=self._debug_mm_euler_call_count,
            tag="mmEuler_label",
        )

    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        input_frames: Optional[torch.Tensor],
        return_detailed: bool = True,
        keep_bc_dims: bool = False,
        preserve_component_grads: bool = False
    ):
        pred = predictions
        pred = self.norm_helper.denormalize(pred)

        label_fields_full: Optional[Dict[str, torch.Tensor]] = None
        if labels is not None:
            labels = self.norm_helper.denormalize(labels)
            label_fields_full = self._tensor_to_fields(labels)

        fields_full = self._tensor_to_fields(pred)

        # build prev_fields from last input frame
        prev_fields = {}
        if input_frames is not None:
            input_frames = self.norm_helper.denormalize(input_frames)
            for i, name in enumerate(self.field_names):
                prev_fields[name] = input_frames[:, -1:, i, ...]   # (B,1,*spatial)

        derivs, residuals = self._compute_component_residuals(
            fields_full=fields_full,
            prev_fields=prev_fields,
            label_fields_full=label_fields_full,
        )

        if (
            self._debug_mm_euler_on_labels
            and label_fields_full is not None
            and self._debug_mm_euler_call_count < self._debug_mm_euler_max_calls
        ):
            label_derivs, label_residuals = self._compute_component_residuals(
                fields_full=label_fields_full,
                prev_fields=prev_fields,
                label_fields_full=label_fields_full,
            )
            mm_debug_tensors = self._collect_mm_debug_tensors(label_derivs, label_residuals)
            self._plot_mm_debug_tensors(mm_debug_tensors)
            self._debug_mm_euler_call_count += 1

        # Stack residuals into channel dim: (B,T_eval,Ceq,*spatial)
        eq_names = list(residuals.keys())
        res_stack = torch.stack([residuals[k] for k in eq_names], dim=2)

        # Apply reference-based scaling if available
        if self.residual_scales:
            scales = [self.residual_scales.get(k, 1.0) for k in eq_names]
            scale_t = torch.tensor(scales, device=res_stack.device, dtype=res_stack.dtype)
            view_shape = [1, 1, len(eq_names)] + [1] * (res_stack.ndim - 3)
            res_stack = res_stack / (scale_t.view(*view_shape) + self.residual_scale_eps)

        # Elementwise penalty on scaled residuals
        pen = robust_penalty(res_stack, kind=self.penalty, huber_delta=self.huber_delta)

        if self.residual_masker is not None:
            pen = self.residual_masker.apply_penalty(pen, eq_names, self.components, derivs)

            if (
                self._debug_residual_mask_on
                and self._debug_residual_mask_call_count < self._debug_residual_mask_max_calls
            ):
                mask_debug_tensors = self._collect_residual_mask_debug_tensors()
                self._plot_debug_tensors(
                    tensors=mask_debug_tensors,
                    output_dir=self._debug_residual_mask_output_dir,
                    sample_index=self._debug_residual_mask_sample_index,
                    time_index=self._debug_residual_mask_time_index,
                    max_plots=self._debug_residual_mask_max_plots,
                    clip_percent=self._debug_residual_mask_clip_percent,
                    run_index=self._debug_residual_mask_call_count,
                    tag="residual_mask",
                )
                self._debug_residual_mask_call_count += 1

        # Reduce over spatial dims -> (B,T_eval,Ceq)
        spatial_dims = list(range(3, pen.ndim))
        pen_red = pen.mean(dim=spatial_dims) if spatial_dims else pen  # (B,T_eval,Ceq)

        # Keep batch and component dims (for rollout metrics)
        if keep_bc_dims:
            per_eq = pen_red.mean(dim=1)  # (B, Ceq)

            if self.weight_schedule.component_weights:
                component_order = list(self.weight_schedule.component_weights.keys())
            else:
                component_order = eq_names

            per_component = []
            for name in component_order:
                if name not in eq_names:
                    raise ValueError(
                        f"PDEResidualLoss: component '{name}' not found in configured equations {eq_names}."
                    )
                idx = eq_names.index(name)
                value = per_eq[:, idx]
                q_weight = self.weight_schedule.get_loss_component_weight(name)
                q_weight_tensor = torch.tensor(
                    q_weight,
                    device=value.device,
                    dtype=value.dtype,
                )
                weighted_value = self.weight_schedule.base_weight * q_weight_tensor * value
                per_component.append(weighted_value)

            per_component_tensor = torch.stack(per_component, dim=1)
            return per_component_tensor

        # Conditionally detach based on preserve_component_grads
        pen_red_for_detailed = pen_red if preserve_component_grads else pen_red.detach()

        # Compute per-equation components (mean over batch and time)
        per_equation = pen_red_for_detailed.mean(dim=(0, 1))  # (Ceq,)

        component_weights = torch.tensor(
            [self.weight_schedule.get_loss_component_weight(n) for n in eq_names],
            device=pen_red.device,
            dtype=pen_red.dtype,
        ).view(1, 1, -1)


        weighted = pen_red * component_weights
        weighted_per_equation = weighted.mean(dim=(0, 1)) * self.weight_schedule.base_weight

        print(weighted)

        all_components = {eq_names[i]: weighted_per_equation[i] for i in range(len(eq_names))}
        loss_weighted = weighted.mean()
        loss_weighted = loss_weighted * self.weight_schedule.base_weight

        if not return_detailed:
            return loss_weighted

        detailed = {
            "per_component": {name: value for name, value in all_components.items()}
        }
        return loss_weighted, detailed

    def _compute_derivatives(
        self,
        fields_full: Dict[str, torch.Tensor],
        prev_fields: Optional[Dict[str, torch.Tensor]],
    ) -> DerivativeCache:
        required_fields = set()
        required_time_fields = set()
        required_grad_fields = set()
        required_laplacian_fields = set()

        for comp in self.components:
            required_fields.update(comp.required_fields)
            required_time_fields.update(comp.required_time_fields)
            required_grad_fields.update(comp.required_grad_fields)
            required_laplacian_fields.update(comp.required_laplacian_fields)

        if self.residual_mask_grad_fields:
            required_grad_fields.update(self.residual_mask_grad_fields)

        all_required = set().union(
            required_fields,
            required_time_fields,
            required_grad_fields,
            required_laplacian_fields,
        )
        available_fields = list(fields_full.keys())
        resolved_field_map: Dict[str, str] = {}
        unresolved: List[str] = []
        for req in sorted(all_required):
            resolved = _resolve_field_name(req, available_fields)
            if resolved is None:
                unresolved.append(req)
            else:
                resolved_field_map[req] = resolved
        if unresolved:
            raise ValueError(
                "PDEResidualLoss: missing required fields: "
                f"{unresolved}. Available fields: {available_fields}"
            )

        t_idx: Optional[torch.Tensor] = None
        time_derivs: Dict[str, torch.Tensor] = {}
        time_derivs_by_resolved: Dict[str, torch.Tensor] = {}

        for field in required_time_fields:
            resolved = resolved_field_map[field]
            if resolved in time_derivs_by_resolved:
                ut = time_derivs_by_resolved[resolved]
                t_field = t_idx if t_idx is not None else None
            else:
                prev = prev_fields.get(resolved) if prev_fields is not None else None
                ut, t_field = self.ops.time_derivative(
                    fields_full[resolved],
                    eval_on=self.eval_time,
                    prev=prev,
                )
                time_derivs_by_resolved[resolved] = ut
            if t_idx is None:
                if t_field is None:
                    raise RuntimeError("Internal error: missing time index while computing derivatives.")
                t_idx = t_field
            elif not torch.equal(t_idx, t_field):
                raise ValueError("PDEResidualLoss: time derivative indices are inconsistent across fields.")
            time_derivs[field] = ut
            time_derivs[resolved] = ut

        if t_idx is None:
            T_full = next(iter(fields_full.values())).shape[1]
            t_idx = torch.arange(0, T_full, device=next(iter(fields_full.values())).device)

        fields_n: Dict[str, torch.Tensor] = {}
        for name in required_fields:
            resolved = resolved_field_map[name]
            value = fields_full[resolved][:, t_idx]
            fields_n[name] = value
            fields_n[resolved] = value

        grads: Dict[str, List[torch.Tensor]] = {}
        grads_by_resolved: Dict[str, List[torch.Tensor]] = {}
        for field in required_grad_fields:
            resolved = resolved_field_map[field]
            if resolved in grads_by_resolved:
                g_eval = grads_by_resolved[resolved]
            else:
                g_full = self.ops.grad(fields_full[resolved])
                g_eval = [g[:, t_idx] for g in g_full]
                grads_by_resolved[resolved] = g_eval
            grads[field] = g_eval
            grads[resolved] = g_eval

        laplacian: Dict[str, torch.Tensor] = {}
        laplacian_by_resolved: Dict[str, torch.Tensor] = {}
        for field in required_laplacian_fields:
            resolved = resolved_field_map[field]
            if resolved in laplacian_by_resolved:
                lap_eval = laplacian_by_resolved[resolved]
            else:
                lap_full = self.ops.laplacian(fields_full[resolved])
                lap_eval = lap_full[:, t_idx]
                laplacian_by_resolved[resolved] = lap_eval
            laplacian[field] = lap_eval
            laplacian[resolved] = lap_eval

        return DerivativeCache(
            t_idx=t_idx,
            fields_n=fields_n,
            time=time_derivs,
            grads=grads,
            laplacian=laplacian,
        )
