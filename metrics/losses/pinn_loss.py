from __future__ import annotations

from dataclasses import dataclass
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
        axis, side = _boundary_axis_from_name(self.boundary, self.spatial_dim)
        axis_index = _physical_to_tensor_axis(axis, self.spatial_dim)

        residuals = []
        for field in self.fields:
            u = fields[field]
            target = self._value_for_field(field, self.value)
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
        if self.axis is None:
            axis_grad, side = _boundary_axis_from_name(self.boundary, self.spatial_dim)
            axis_mask = axis_grad
        else:
            axis_grad = int(self.axis)
            if self.boundary is not None:
                axis_mask, side = _boundary_axis_from_name(self.boundary, self.spatial_dim)
            else:
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
                "PINNLoss: equation_weight masking requires Velocity_X and Velocity_Y gradients."
            )
        du_dx, _ = derivs.grads["Velocity_X"]
        _, dv_dy = derivs.grads["Velocity_Y"]
        div_u = du_dx + dv_dy

        denom = self.k1 * (div_u.abs() - div_u) + 1.0
        eq_weight = 1.0 / (denom + 1e-12)
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
        for field_name, alpha_i, beta_i in zip(self.fields, self.alpha, self.beta):
            if field_name not in derivs.grads:
                raise ValueError(
                    f"PINNLoss: lambda1_grad masking requires gradients for '{field_name}'."
                )
            grad_components = derivs.grads[field_name]
            grad_stack = torch.stack(grad_components, dim=0)
            g = torch.sqrt((grad_stack ** 2).sum(dim=0) + self.eps)
            weights.append(1.0 / (1.0 + alpha_i * (g ** beta_i)))

        lam = torch.stack(weights, dim=0).mean(dim=0)
        lam = lam.clamp_min(self.eps)

        factors = []
        for comp in components:
            factors.append(torch.ones_like(lam) if isinstance(comp, _BoundaryConditionBase) else lam)
        factor_t = torch.stack(factors, dim=2)
        return penalty * factor_t


_RESIDUAL_MASK_REGISTRY: Dict[str, Callable[..., ResidualMasker]] = {
    "EquationWeight": EquationWeight,
    "GradientAnnihilated": GradientAnnihilated,
}


# ----------------------------
# The LossComponent: PINN residual metric for AR rollouts
# ----------------------------

class PINNLoss(LossComponent):
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
        pde_params: Optional[Dict[str, float]] = None,
        reference_quantities: Optional[Dict[str, float]] = None,
        residual_scale_eps: float = 1e-8,
    ):
        super().__init__(
            norm_helper=norm_helper,
            weight=weight,
            name=name or "PINNLoss",
            data_dim=data_dim,
            field_names=field_names,
        )
        if field_names is None:
            raise ValueError("PINNLoss requires field_names to map channels to PDE fields.")

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
                fields_cfg = mask_cfg.get("lambda1_fields", lambda1_fields)
                if alpha_cfg is None or beta_cfg is None:
                    raise ValueError(
                        "PINNLoss: GradientAnnihilated masking requires lambda1_alpha and lambda1_beta."
                    )
                fields = fields_cfg or list(self.field_names)
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
                    "PINNLoss: component weights must match configured components. "
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
                raise ValueError("PINNLoss requires 'components' configuration.")
            if isinstance(pde, str) and pde == "EulerPrimitives2D":
                components = [
                    {"type": "pde/unsteadyContinuity"},
                    {"type": "pde/eulerMomentumX"},
                    {"type": "pde/eulerMomentumY"},
                ]
            elif isinstance(pde, str) and pde in ("debugsystem", "DebugSystem2D"):
                components = [
                    {"type": "pde/debugSystemEq1"},
                    {"type": "pde/debugSystemEq2"},
                ]
            else:
                raise ValueError(
                    "PINNLoss legacy 'pde' config is unsupported. "
                    "Provide a 'components' list instead."
                )

        if not isinstance(components, list) or len(components) == 0:
            raise ValueError("PINNLoss: 'components' must be a non-empty list.")

        component_instances: List[PDEComponent] = []
        component_names: List[str] = []
        component_weights: Dict[str, float] = {}
        component_name_set: set = set()

        for comp in components:
            if "type" not in comp:
                raise ValueError("PINNLoss: each component must have a 'type'.")
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
                            "PINNLoss: boundary condition components require field_names."
                        )
                    cfg["field_names"] = field_names
            if comp_type == "pde/eulerMomentum" and "direction" in cfg:
                raise ValueError(
                    "PINNLoss: 'pde/eulerMomentum' no longer accepts a 'direction' field. "
                    "Use 'pde/eulerMomentumX' or 'pde/eulerMomentumY' instead."
                )
            name = cfg.pop("name", None)
            if "spatial_dim" not in cfg and data_dim is not None:
                cfg = {**cfg, "spatial_dim": data_dim}
            comp_instance = _PDE_COMPONENT_REGISTRY[comp_type](name=name, **cfg)
            if comp_instance.name in component_name_set:
                raise ValueError(
                    "PINNLoss: duplicate component name '"
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

        fields_full = self._tensor_to_fields(pred)

        # build prev_fields from last input frame
        prev_fields = {}
        if input_frames is not None:
            input_frames = self.norm_helper.denormalize(input_frames)
            for i, name in enumerate(self.field_names):
                prev_fields[name] = input_frames[:, -1:, i, ...]   # (B,1,*spatial)

        pde_params = {**self.pde_params, "eval_time": self.eval_time}
        derivs = self._compute_derivatives(fields_full, prev_fields)

        residuals: Dict[str, torch.Tensor] = {}
        for comp in self.components:
            res = comp.residual(derivs.fields_n, derivs, params={**comp.params, **pde_params})
            residuals[comp.name] = res

        if self.residual_masker is not None:
            residuals = self.residual_masker.apply_residuals(residuals, self.components, derivs)

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
                        f"PINNLoss: component '{name}' not found in configured equations {eq_names}."
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
        all_components = {eq_names[i]: per_equation[i] for i in range(len(eq_names))}

        component_weights = torch.tensor(
            [self.weight_schedule.get_loss_component_weight(n) for n in eq_names],
            device=pen_red.device,
            dtype=pen_red.dtype,
        ).view(1, 1, -1)

        weighted = pen_red * component_weights
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

        missing = [f for f in required_fields if f not in fields_full]
        if missing:
            raise ValueError(f"PINNLoss: missing required fields: {missing}")

        t_idx: Optional[torch.Tensor] = None
        time_derivs: Dict[str, torch.Tensor] = {}

        for field in required_time_fields:
            prev = prev_fields.get(field) if prev_fields is not None else None
            ut, t_field = self.ops.time_derivative(
                fields_full[field],
                eval_on=self.eval_time,
                prev=prev,
            )
            if t_idx is None:
                t_idx = t_field
            elif not torch.equal(t_idx, t_field):
                raise ValueError("PINNLoss: time derivative indices are inconsistent across fields.")
            time_derivs[field] = ut

        if t_idx is None:
            T_full = next(iter(fields_full.values())).shape[1]
            t_idx = torch.arange(0, T_full, device=next(iter(fields_full.values())).device)

        fields_n = {name: fields_full[name][:, t_idx] for name in required_fields}

        grads: Dict[str, List[torch.Tensor]] = {}
        for field in required_grad_fields:
            g_full = self.ops.grad(fields_full[field])
            grads[field] = [g[:, t_idx] for g in g_full]

        laplacian: Dict[str, torch.Tensor] = {}
        for field in required_laplacian_fields:
            lap_full = self.ops.laplacian(fields_full[field])
            laplacian[field] = lap_full[:, t_idx]

        return DerivativeCache(
            t_idx=t_idx,
            fields_n=fields_n,
            time=time_derivs,
            grads=grads,
            laplacian=laplacian,
        )
