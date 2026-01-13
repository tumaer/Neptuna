from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable, Literal, Any
from ..loss_framework import LossComponent, WeightSchedule, apply_batch_wise_normalization, NormalizationHelper

import torch
import torch.nn as nn

# PDE registry local to this file
_PDE_REGISTRY: Dict[str, Callable[..., "PDESystem"]] = {}

# Spatial backend registry local to this file
_SPATIAL_BACKEND_REGISTRY: Dict[str, Callable[..., "SpatialDerivativeBackend"]] = {}

def register_pde(name: str):
    def deco(fn):
        _PDE_REGISTRY[name] = fn
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
# PDE interface (modular)
# ----------------------------

class PDESystem(nn.Module):
    """
    A modular PDE system that produces a dict of residual components.

    residuals must be shaped:
      (B, T_eval, *spatial) for each equation, OR
      (B, T_eval, C_eq, *spatial) if you prefer a stacked form.

    We'll standardize to dict[str -> (B,T_eval,*spatial)] for clarity.
    """
    equation_names: List[str]

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        *,
        ops: "DifferentialOps",
        params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


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


@register_pde("EulerPrimitives2D")
class EulerPrimitives2D(PDESystem):
    """
    Compressible Euler (barotropic/primitive form) in 2D with fields:
      Density, Pressure, Velocity_X, Velocity_Y.
    """
    def __init__(self, spatial_dim: int = 2):
        super().__init__()
        if spatial_dim != 2:
            raise ValueError(f"EulerPrimitives2D only supports spatial_dim=2, got {spatial_dim}")
        self.spatial_dim = spatial_dim
        self.equation_names = ["continuity", "momentum_x", "momentum_y"]
    
    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        *,
        ops: DifferentialOps,
        params: Optional[Dict[str, float]] = None,
        prev_fields: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        eval_time = params.get("eval_time", "interior") if params else "interior"

        rho = fields["Density"]
        p   = fields["Pressure"]
        u   = fields["Velocity_X"]
        v   = fields["Velocity_Y"]

        prev_rho = prev_p = prev_u = prev_v = None
        if prev_fields is not None:
            prev_rho = prev_fields.get("Density")
            prev_p   = prev_fields.get("Pressure")
            prev_u   = prev_fields.get("Velocity_X")
            prev_v   = prev_fields.get("Velocity_Y")

        mom_x = rho * u
        mom_y = rho * v
        prev_mom_x = prev_mom_y = None
        if prev_rho is not None and prev_u is not None and prev_v is not None:
            prev_mom_x = prev_rho * prev_u
            prev_mom_y = prev_rho * prev_v

        rho_t, t_idx = ops.time_derivative(rho, eval_on=eval_time, prev=prev_rho)
        momx_t, _    = ops.time_derivative(mom_x, eval_on=eval_time, prev=prev_mom_x)
        momy_t, _    = ops.time_derivative(mom_y, eval_on=eval_time, prev=prev_mom_y)

        rho_n = rho[:, t_idx]
        p_n   = p[:, t_idx]
        u_n   = u[:, t_idx]
        v_n   = v[:, t_idx]
        momx_n = mom_x[:, t_idx]
        momy_n = mom_y[:, t_idx]

        flux_rho = [rho_n * u_n, rho_n * v_n]
        flux_mx  = [rho_n * u_n * u_n + p_n, rho_n * u_n * v_n]
        flux_my  = [rho_n * u_n * v_n,       rho_n * v_n * v_n + p_n]

        cont   = rho_t + ops.div(flux_rho)
        mx_res = momx_t + ops.div(flux_mx)
        my_res = momy_t + ops.div(flux_my)

        return {"continuity": cont, "momentum_x": mx_res, "momentum_y": my_res}


@register_pde("TwoPhaseProxyEulerPrimitives2D")
class TwoPhaseProxyEulerPrimitives2D(PDESystem):
    """
    Proxy inviscid compressible continuity + momentum residuals in 2D primitive variables.

    Intended for datasets like LIDE/SIDA where you have mixture-like:
      - Density
      - Pressure
      - Velocity_X
      - Velocity_Y

    Notes:
      * This is NOT the full multiphase Euler used by ALPACA (no alpha, no nonconservative terms).
      * Use this as a baseline "physics consistency" residual.
      * Supports optional axisymmetric geometry corrections via params (see below).

    Params (optional):
      - eval_time: "interior" (default) or "all"
      - geometry: "planar" (default) or "axisymmetric"
      - r_coord:  Tensor broadcastable to spatial grid (for axisymmetric), shape (1,1,H,W) or (B,T,H,W)
      - eps_r: small epsilon to avoid division by zero at r=0 (default 1e-6)
    """
    def __init__(self, spatial_dim: int = 2):
        super().__init__()
        if spatial_dim != 2:
            raise ValueError(f"TwoPhaseProxyEulerPrimitives2D only supports spatial_dim=2, got {spatial_dim}")
        self.spatial_dim = 2
        self.equation_names = ["continuity", "momentum_x", "momentum_y"]

    def _axisym_divergence(
        self,
        ops,
        flux_x: torch.Tensor,
        flux_y: torch.Tensor,
        *,
        r_coord: torch.Tensor,
        eps_r: float,
        flux_x_is_radial: bool = True,
    ) -> torch.Tensor:
        """
        Axisymmetric divergence for 2D (r,z) data:
          div(F) = (1/r) d_r(r F_r) + d_z(F_z)

        We assume:
          - x-direction corresponds to radial r (Velocity_X = u_r)
          - y-direction corresponds to axial z (Velocity_Y = u_z)

        If your dataset uses different conventions, swap flux components accordingly.
        """
        # r*F_r
        r = r_coord.clamp_min(eps_r)
        rFr = r * flux_x if flux_x_is_radial else flux_x

        # d_r(r F_r)
        dr_rFr = ops.grad(rFr)[0]  # grad returns [d/dx, d/dy] in planar interpretation
        # d_z(F_z)
        dz_Fz = ops.grad(flux_y)[1]

        return dr_rFr / r + dz_Fz

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        *,
        ops,
        params: Optional[Dict[str, float]] = None,
        prev_fields: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        params = params or {}
        eval_time = params.get("eval_time", "interior")
        geometry = params.get("geometry", "planar")
        eps_r = float(params.get("eps_r", 1e-6))

        rho = fields["Density"]      # (B,T,H,W)
        p   = fields["Pressure"]     # (B,T,H,W)
        u   = fields["Velocity_X"]   # (B,T,H,W)  (assumed u_r for axisymmetric)
        v   = fields["Velocity_Y"]   # (B,T,H,W)  (assumed u_z for axisymmetric)

        prev_rho = prev_p = prev_u = prev_v = None
        if prev_fields is not None:
            prev_rho = prev_fields.get("Density")
            prev_p   = prev_fields.get("Pressure")
            prev_u   = prev_fields.get("Velocity_X")
            prev_v   = prev_fields.get("Velocity_Y")

        # Conservative momenta
        mom_x = rho * u
        mom_y = rho * v
        prev_mom_x = prev_mom_y = None
        if prev_rho is not None and prev_u is not None and prev_v is not None:
            prev_mom_x = prev_rho * prev_u
            prev_mom_y = prev_rho * prev_v

        # Time derivatives (FD in time via ops)
        rho_t, t_idx = ops.time_derivative(rho, eval_on=eval_time, prev=prev_rho)
        momx_t, _    = ops.time_derivative(mom_x, eval_on=eval_time, prev=prev_mom_x)
        momy_t, _    = ops.time_derivative(mom_y, eval_on=eval_time, prev=prev_mom_y)

        # Align fields to evaluated time indices
        rho_n = rho[:, t_idx]
        p_n   = p[:, t_idx]
        u_n   = u[:, t_idx]
        v_n   = v[:, t_idx]
        momx_n = mom_x[:, t_idx]
        momy_n = mom_y[:, t_idx]

        # Fluxes (planar form)
        # Continuity: d_t rho + div(rho u) = 0
        flux_rho_x = rho_n * u_n
        flux_rho_y = rho_n * v_n

        # Momentum (inviscid, conservative):
        # d_t(rho u) + div([rho u^2 + p, rho u v]) = 0
        # d_t(rho v) + div([rho u v, rho v^2 + p]) = 0
        flux_mx_x  = rho_n * u_n * u_n + p_n
        flux_mx_y  = rho_n * u_n * v_n
        flux_my_x  = rho_n * u_n * v_n
        flux_my_y  = rho_n * v_n * v_n + p_n

        if geometry == "planar":
            cont   = rho_t  + ops.div([flux_rho_x, flux_rho_y])
            mx_res = momx_t + ops.div([flux_mx_x,  flux_mx_y])
            my_res = momy_t + ops.div([flux_my_x,  flux_my_y])

        elif geometry == "axisymmetric":
            # Requires r_coord to compute (1/r) d_r(r F_r)
            if "r_coord" not in params:
                raise ValueError(
                    "Axisymmetric geometry requires params['r_coord'] broadcastable to (B,T_eval,H,W) "
                    "or (1,1,H,W)."
                )
            r_coord = params["r_coord"]
            # Make r_coord time-aligned if needed
            if r_coord.ndim == 4:
                # (B,T,H,W) or (1,1,H,W) -> slice time if it has T dimension
                if r_coord.shape[1] == rho.shape[1]:
                    r_coord = r_coord[:, t_idx]
                elif r_coord.shape[1] in (1,):  # constant in time
                    r_coord = r_coord.expand(rho_n.shape[0], rho_n.shape[1], *r_coord.shape[-2:])
            elif r_coord.ndim == 2:
                # (H,W) -> (1,1,H,W)
                r_coord = r_coord[None, None, ...].expand(rho_n.shape[0], rho_n.shape[1], *r_coord.shape)
            else:
                raise ValueError(f"Unsupported r_coord shape: {tuple(r_coord.shape)}")

            cont   = rho_t  + self._axisym_divergence(ops, flux_rho_x, flux_rho_y, r_coord=r_coord, eps_r=eps_r)
            mx_res = momx_t + self._axisym_divergence(ops, flux_mx_x,  flux_mx_y,  r_coord=r_coord, eps_r=eps_r)
            my_res = momy_t + self._axisym_divergence(ops, flux_my_x,  flux_my_y,  r_coord=r_coord, eps_r=eps_r)
        else:
            raise ValueError(f"Unknown geometry: {geometry} (expected 'planar' or 'axisymmetric')")

        return {
            "continuity": cont,
            "momentum_x": mx_res,
            "momentum_y": my_res,
        }

@register_pde("VorticityConsistency2D")
class VorticityConsistency2D(PDESystem):
    """
    Vorticity definition consistency residual in 2D.

    Expects fields:
      - Velocity_X : u_r (radial component)
      - Velocity_Y : u_z (axial component)
      - Vorticity  : omega_theta (out-of-plane vorticity for axisymmetric r-z slice)

    Residual:
      R_omega = omega - (curl u)

    For planar 2D (x,y):
      curl(u) = d/dx(v) - d/dy(u)

    For axisymmetric (r,z), the theta-component of vorticity is:
      omega_theta = d/dr(u_z) - d/dz(u_r)

    With your convention:
      Velocity_X = u_r  (radial, x)
      Velocity_Y = u_z  (axial,  y)
      axis is at x=0
    we use:
      curl = d_x(Velocity_Y) - d_y(Velocity_X)

    Params (optional):
      - geometry: "planar" (default) or "axisymmetric"
        (for this residual the formula is the same with the stated conventions)
      - eval_time: "interior" or "all"  (vorticity residual is time-local; we just slice to match)
    """
    def __init__(self, spatial_dim: int = 2):
        super().__init__()
        if spatial_dim != 2:
            raise ValueError(f"VorticityConsistency2D only supports spatial_dim=2, got {spatial_dim}")
        self.spatial_dim = 2
        self.equation_names = ["vorticity_consistency"]

    def residual(
        self,
        fields: Dict[str, torch.Tensor],
        *,
        ops,
        params: Optional[Dict[str, float]] = None,
        prev_fields: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        params = params or {}
        eval_time = params.get("eval_time", "interior")

        u_r = fields["Velocity_X"]   # (B,T,H,W)
        u_z = fields["Velocity_Y"]   # (B,T,H,W)
        omega = fields["Vorticity"]  # (B,T,H,W)

        # Align time indexing with other PDE systems for consistent logging/aggregation.
        # Since this is purely spatial, we just pick the same t_idx you would use for FD time residuals:
        # - "interior": drop endpoints (matches center2 time derivative use elsewhere)
        # - "all": keep all times
        T = u_r.shape[1]
        if eval_time == "interior":
            if T < 3:
                raise ValueError("Need T>=3 for eval_time='interior' time alignment.")
            t_idx = torch.arange(1, T - 1, device=u_r.device, dtype=torch.long)
        elif eval_time == "all":
            t_idx = torch.arange(0, T, device=u_r.device, dtype=torch.long)
        else:
            raise ValueError(f"Unknown eval_time: {eval_time} (expected 'interior' or 'all')")

        u_r_n = u_r[:, t_idx]
        u_z_n = u_z[:, t_idx]
        omega_n = omega[:, t_idx]

        # Spatial derivatives
        # grad returns [d/dx, d/dy] where x is radial and y is axial in your convention.
        duz_dx = ops.grad(u_z_n)[0]  # d(u_z)/dr
        dur_dy = ops.grad(u_r_n)[1]  # d(u_r)/dz

        curl_theta = duz_dx - dur_dy
        res = omega_n - curl_theta

        return {"vorticity_consistency": res}

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


# ----------------------------
# The LossComponent: PINN residual metric for AR rollouts
# ----------------------------

class PINNLoss(LossComponent):
    """
    Strong-form residual loss for grid/rollout predictions.
    """
    def __init__(
        self,
        pde: Union[str, PDESystem, Dict[str, Union[str, float, int]]],
        *,
        dt: float,
        dx: Union[float, Tuple[float, ...]],
        spatial_backend: Union[str, SpatialDerivativeBackend, Dict[str, Any]],
        time_scheme: Literal["forward1", "center2"] = "forward1",
        eval_time: Literal["interior", "all"] = "interior",
        # weighting/aggregation
        time_aggregation: Literal["mean", "tail_mean", "exp"] = "mean",
        tail_fraction: float = 0.2,
        exp_gamma: float = 2.0,
        # residual penalty
        penalty: Literal["l2", "huber"] = "l2",
        huber_delta: float = 1.0,
        # optional masking (e.g., shock/interface mask)
        mask_channel: Optional[str] = None,
        mask_mode: Literal["multiply", "exclude"] = "multiply",
        # normalization of residual magnitudes
        residual_normalization: Literal["none", "magnitude", "variance"] = "none",
        # framework bits
        norm_helper: Optional["NormalizationHelper"] = None,
        weight: Union[float, "WeightSchedule"] = 1.0,
        name: Optional[str] = None,
        data_dim: Optional[int] = None,
        field_names: Optional[List[str]] = None,
        pde_params: Optional[Dict[str, float]] = None,
        compute_in_physical_space: bool = True,
        reference_quantities: Optional[Dict[str, float]] = None,
        residual_scale_eps: float = 1e-8,
    ):
        super().__init__(
            norm_helper=norm_helper,
            weight=weight,
            name=name or "PINNResidual",
            data_dim=data_dim,
            field_names=field_names,
        )
        if field_names is None:
            raise ValueError("PINNLoss requires field_names to map channels to PDE fields.")

        self.pde = self._build_pde(pde, data_dim)
        self.ops = DifferentialOps(
            dt=float(dt),
            dx=dx,
            spatial_backend=self._build_spatial_backend(spatial_backend, data_dim),
            time_scheme=time_scheme,
        )
        self.eval_time = eval_time
        self.time_aggregation = time_aggregation
        self.tail_fraction = float(tail_fraction)
        self.exp_gamma = float(exp_gamma)

        self.penalty = penalty
        self.huber_delta = float(huber_delta)

        self.mask_channel = mask_channel
        self.mask_mode = mask_mode

        self.residual_normalization = residual_normalization
        self.pde_params = pde_params or {}
        self.compute_in_physical_space = compute_in_physical_space

        self.reference_quantities = reference_quantities or {}
        self.residual_scale_eps = float(residual_scale_eps)
        self.residual_scales = self._build_residual_scales(self.pde, self.reference_quantities)

    @staticmethod
    def _build_residual_scales(pde: PDESystem, refs: Dict[str, float]) -> Dict[str, float]:
        """
        Build characteristic scales for each PDE residual to make them O(1).
        Currently implemented for EulerPrimitives2D; falls back to 1.0 otherwise.
        """
        if isinstance(pde, EulerPrimitives2D):
            ref_rho = refs.get("density", 1.0)
            ref_u = refs.get("velocity", 1.0)
            ref_p = refs.get("pressure", 1.0)
            ref_L = refs.get("length", 1.0)
            eps = 1e-12
            ref_t = ref_L / max(ref_u, eps)

            # continuity: rho_t + div(rho u)
            cont_scale = max(ref_rho / ref_t, ref_rho * ref_u / ref_L)

            # momentum: (rho u)_t + div(rho u u + p I)
            mom_scale = max(
                ref_rho * ref_u / ref_t,
                ref_rho * ref_u * ref_u / ref_L,
                ref_p / ref_L,
            )
            return {
                "continuity": cont_scale,
                "momentum_x": mom_scale,
                "momentum_y": mom_scale,
            }
        # default: no scaling
        return {}
    
    @staticmethod
    def _build_pde(pde_spec: Union[str, PDESystem, Dict[str, Union[str, float, int]]], data_dim: Optional[int]) -> PDESystem:
        if isinstance(pde_spec, PDESystem):
            return pde_spec
        if isinstance(pde_spec, str):
            name = pde_spec
            cfg: Dict[str, Union[str, float, int]] = {}
        elif isinstance(pde_spec, dict):
            name = pde_spec.get("type", None)
            if name is None:
                raise ValueError("PDE dict must contain a 'type' field.")
            cfg = {k: v for k, v in pde_spec.items() if k != "type"}
        else:
            raise ValueError(f"Unsupported pde spec type: {type(pde_spec)}")

        if name not in _PDE_REGISTRY:
            raise ValueError(f"Unknown PDE type '{name}'. Available: {list(_PDE_REGISTRY.keys())}")
        # Provide data_dim as default spatial_dim if not set
        if "spatial_dim" not in cfg and data_dim is not None:
            cfg = {**cfg, "spatial_dim": data_dim}
        return _PDE_REGISTRY[name](**cfg)

    def _tensor_to_fields(self, pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pred.shape[2] != len(self.field_names):
            raise ValueError(
                f"pred has C={pred.shape[2]} but field_names has {len(self.field_names)} entries."
            )
        return {name: pred[:, :, i, ...] for i, name in enumerate(self.field_names)}

    def _get_mask(self, fields: Dict[str, torch.Tensor], t_idx: torch.Tensor) -> Optional[torch.Tensor]:
        if self.mask_channel is None:
            return None
        if self.mask_channel not in fields:
            raise ValueError(f"mask_channel={self.mask_channel} not found in fields: {list(fields.keys())}")
        m = fields[self.mask_channel]  # (B,T,*spatial)
        return m[:, t_idx]             # (B,T_eval,*spatial)

    def _aggregate_time(self, per_t: torch.Tensor) -> torch.Tensor:
        if self.time_aggregation == "mean":
            return per_t.mean()
        T = per_t.numel()
        if T == 0:
            return per_t.new_tensor(0.0)
        if self.time_aggregation == "tail_mean":
            k = max(1, int(round(T * self.tail_fraction)))
            return per_t[-k:].mean()
        if self.time_aggregation == "exp":
            w = torch.linspace(0.0, 1.0, steps=T, device=per_t.device, dtype=per_t.dtype)
            w = torch.exp(self.exp_gamma * w)
            w = w / (w.sum() + 1e-12)
            return (per_t * w).sum()
        raise ValueError(f"Unknown time_aggregation: {self.time_aggregation}")

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
    ):
        pred = predictions
        if self.compute_in_physical_space and self.norm_helper is not None:
            pred = self.norm_helper.denormalize(pred)

        fields_full = self._tensor_to_fields(pred)

        # build prev_fields from last input frame
        prev_fields = None
        if input_frames is not None and input_frames.shape[0] == predictions.shape[0] and input_frames.shape[1] >= 1:
            prev_fields = {}
            if self.compute_in_physical_space and self.norm_helper is not None:
                input_frames = self.norm_helper.denormalize(input_frames)
            for i, name in enumerate(self.field_names):
                prev_fields[name] = input_frames[:, -1:, i, ...]   # (B,1,*spatial)

        pde_params = {**self.pde_params, "eval_time": self.eval_time}
        residuals = self.pde.residual(fields_full, ops=self.ops, params=pde_params, prev_fields=prev_fields)

        # Standardize evaluation time indices by inspecting one residual
        any_key = next(iter(residuals.keys()))
        res0 = residuals[any_key]
        if res0.ndim < 3:
            raise ValueError("Residual tensors must have shape (B, T_eval, *spatial).")
        T_eval = res0.shape[1]

        # Stack residuals into channel dim: (B,T_eval,Ceq,*spatial)
        eq_names = list(residuals.keys())
        res_stack = torch.stack([residuals[k] for k in eq_names], dim=2)

        # Apply reference-based scaling if available
        if self.residual_scales:
            scales = [self.residual_scales.get(k, 1.0) for k in eq_names]
            scale_t = torch.tensor(scales, device=res_stack.device, dtype=res_stack.dtype)
            view_shape = [1, 1, len(eq_names)] + [1] * (res_stack.ndim - 3)
            res_stack = res_stack / (scale_t.view(*view_shape) + self.residual_scale_eps)

        # Infer time indices for masking (best-effort)
        T_full = predictions.shape[1]
        if T_eval == T_full - 2:
            t_idx = torch.arange(1, T_full - 1, device=pred.device)
        elif T_eval == T_full:
            t_idx = torch.arange(0, T_full, device=pred.device)
        elif T_eval == T_full - 1:
            t_idx = torch.arange(0, T_full - 1, device=pred.device)
        else:
            t_idx = torch.arange(0, T_eval, device=pred.device)

        mask = self._get_mask(fields_full, t_idx)  # (B,T_eval,*spatial) or None

        # Elementwise penalty on scaled residuals
        pen = robust_penalty(res_stack, kind=self.penalty, huber_delta=self.huber_delta)

        # Reduce over spatial dims -> (B,T_eval,Ceq), honoring mask if exclude
        spatial_dims = list(range(3, pen.ndim))
        if mask is not None and self.mask_mode == "exclude":
            weights = mask.unsqueeze(2)  # (B,T_eval,1,*spatial)
            num = (pen * weights).sum(dim=spatial_dims)
            denom = weights.sum(dim=spatial_dims).clamp_min(1e-8)
            pen_red = num / denom
        else:
            if mask is not None:  # multiply mode
                pen = pen * mask.unsqueeze(2)
            pen_red = pen.mean(dim=spatial_dims) if spatial_dims else pen  # (B,T_eval,Ceq)

        # Optional residual normalization
        if self.residual_normalization != "none":
            pen_red = apply_batch_wise_normalization(
                pen_red,
                pen_red.detach(),
                normalization=self.residual_normalization,
            )

        # WeightSchedule expects (B,T,C,...) so Ceq acts as channel
        if self.weight_schedule.is_scalar_only():
            # fast path: aggregate over time as configured, then apply base weight
            per_t = pen_red.mean(dim=(0, 2))  # (T_eval,)
            loss_unweighted = (
                self._aggregate_time(per_t)
                if self.time_aggregation != "mean"
                else pen_red.mean()
            )
            loss = loss_unweighted * self.weight_schedule.base_weight
            if not return_detailed:
                return loss
            detailed = {
                "unweighted": loss_unweighted.detach(),
                "equations": eq_names,
                "per_timestep": per_t.detach(),
                "per_channel": pen_red.mean(dim=(0, 1)).detach(),
            }
            return loss, detailed

        # full schedule path (per-timestep/channel weighting), uses mean over time/channels
        w = self.weight_schedule.get_loss_weight(pen_red.shape).to(pen_red.device)  # (1,T_eval,Ceq)
        weighted = pen_red * w
        loss_unweighted = pen_red.mean()
        loss_weighted = weighted.mean() * self.weight_schedule.base_weight

        if not return_detailed:
            return loss_weighted

        detailed = {
            "unweighted": loss_unweighted.detach(),
            "equations": eq_names,
            "per_timestep": pen_red.mean(dim=(0, 2)).detach(),
            "per_channel": pen_red.mean(dim=(0, 1)).detach(),
            "per_timestep_weighted": weighted.mean(dim=(0, 2)).detach(),
            "per_channel_weighted": weighted.mean(dim=(0, 1)).detach(),
        }
        return loss_weighted, detailed


# ----------------------------
# Usage sketch (benchmark config style)
# ----------------------------

"""
# Suppose predictions are (B,T,C,H,W) with channels:
field_names = ["rho", "mom_0", "mom_1", "E", "shock_mask"]  # example, include optional mask
pde = CompressibleEuler(spatial_dim=2, gamma=1.4)

loss = PINNLoss(
    pde=pde,
    dt=dataset_dt,
    dx=(dx, dy),
    spatial_backend=FDSpatialDerivatives(),     # or a spectral backend you implement
    time_scheme="center2",
    eval_time="interior",
    penalty="huber",
    huber_delta=1.0,
    mask_channel="shock_mask",
    mask_mode="multiply",
    residual_normalization="none",
    weight=WeightSchedule(base_weight=1.0, timestep_weights=None, channel_weights=None),
    norm_helper=norm_helper,
    field_names=field_names,
    name="pde_residual",
)
"""
