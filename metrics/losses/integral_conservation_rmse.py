from typing import Literal, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
from torch import nn
from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper

# Inspired by cRMSE and bRMSE from the paper by Takamoto et al.,
# 'PDEBENCH: An Extensive Benchmark for Scientific Machine Learning'
# https://arxiv.org/abs/2210.07182

@dataclass
class BoundaryPatch:
    """Describes a boundary face on a regular Cartesian grid."""
    name: str           # e.g. "west", "east", "south", "north"
    axis: int           # spatial axis: 0 (x), 1 (y), 2 (z)
    side: str           # "min" or "max"
    normal_sign: float  # -1.0 for min face, +1.0 for max face

class DomainQuantity(ABC):
    """
    Base class for domain-integrated conserved quantities.

    Computes spatial integrals of the form: Q = ∫ q(x) dV
    """

    def __init__(self, name: str, required_fields: Sequence[str]):
        self.name = name
        self.required_fields = tuple(required_fields)

    @abstractmethod
    def __call__(
        self,
        fields: Dict[str, torch.Tensor],
        dv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute domain-integrated quantity.

        Parameters
        ----------
        fields : dict of str -> Tensor
            Each tensor has shape (B, T, *spatial).
        dv : Tensor
            Scalar cell volume.

        Returns
        -------
        Tensor of shape (B, T)
            Integrated quantity over space.
        """
        ...

class BoundaryFluxQuantity(ABC):
    """
    Base class for boundary fluxes of conserved quantities.

    Computes surface integrals of the form: Φ = ∫_∂Ω F·n dS
    """

    def __init__(self, name: str, required_fields: Sequence[str]):
        self.name = name
        self.required_fields = tuple(required_fields)

    @abstractmethod
    def __call__(
        self,
        fields: Dict[str, torch.Tensor],
        patches: Sequence[BoundaryPatch],
        spacings: Sequence[float],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute boundary fluxes.

        Parameters
        ----------
        fields : dict of str -> Tensor
            Each tensor has shape (B, T, *spatial).
        patches : list of BoundaryPatch
            Boundary faces to compute fluxes on.
        spacings : sequence of float
            Grid spacings [dx, dy, dz].

        Returns
        -------
        dict of str -> Tensor
            Maps patch_name to flux time series of shape (B, T).
        """
        ...


class IntegralConservationRMSE(LossComponent):
    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: Union[float, WeightSchedule] = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        conserved_keys: Optional[Sequence[str]] = None,
        boundary_keys: Optional[Sequence[str]] = None,
        use_boundary_fluxes: bool = False,
        quantity_weights: Optional[Dict[str, float]] = None,
        components: Optional[List[Dict[str, Union[str, float]]]] = None,
        normalization: Literal['none', 'range', 'variance', 'std'] = 'none',
        eps: float = 1e-8,
    ):
        if components is not None and (conserved_keys or boundary_keys or quantity_weights):
            raise ValueError(
                "IntegralConservationRMSE: specify either 'components' or the "
                "legacy conserved_keys/boundary_keys/quantity_weights, not both."
            )

        component_weights: Optional[Dict[str, float]] = None
        component_names: Optional[List[str]] = None
        boundary_patch_whitelist: Optional[Dict[str, Tuple[str, ...]]] = None

        if components is not None:
            if not isinstance(components, list) or len(components) == 0:
                raise ValueError("IntegralConservationRMSE: 'components' must be a non-empty list.")

            d = data_dim if data_dim is not None else 2
            axis_labels = _axis_labels_from_dim(d)
            valid_patches = {
                _patch_name(a, side) for a in axis_labels for side in ("min", "max")
            }

            component_weights = {}
            component_names = []
            domain_keys: List[str] = []
            boundary_keys_list: List[str] = []
            boundary_patch_map: Dict[str, List[str]] = {}

            for comp in components:
                if "name" not in comp:
                    raise ValueError("IntegralConservationRMSE: each component must have a 'name'.")
                comp_name = str(comp["name"])
                comp_weight = float(comp.get("weight", 1.0))

                parts = comp_name.split("/")
                if parts[0] == "domain" and len(parts) == 2:
                    q_key = parts[1]
                    if q_key not in domain_keys:
                        domain_keys.append(q_key)
                elif parts[0] == "boundary" and len(parts) == 3:
                    if not use_boundary_fluxes:
                        raise ValueError(
                            f"IntegralConservationRMSE: boundary component '{comp_name}' requires "
                            "use_boundary_fluxes=True."
                        )
                    q_key = parts[1]
                    patch = parts[2]
                    if patch not in valid_patches:
                        raise ValueError(
                            f"IntegralConservationRMSE: invalid boundary patch '{patch}' "
                            f"for data_dim={d}. Valid: {sorted(valid_patches)}"
                        )
                    if q_key not in boundary_keys_list:
                        boundary_keys_list.append(q_key)
                    boundary_patch_map.setdefault(q_key, []).append(patch)
                else:
                    raise ValueError(
                        "IntegralConservationRMSE: component name must be 'domain/<key>' or "
                        "'boundary/<key>/<patch>'."
                    )

                component_weights[comp_name] = comp_weight
                component_names.append(comp_name)

            conserved_keys = domain_keys
            boundary_keys = boundary_keys_list
            boundary_patch_whitelist = {
                k: tuple(v) for k, v in boundary_patch_map.items()
            }

        # Convert quantity_weights to WeightSchedule format
        if component_weights is not None:
            quantity_weights = component_weights

        if isinstance(weight, (int, float)):
            weight = WeightSchedule(
                base_weight=float(weight),
                component_weights=quantity_weights
            )
        elif isinstance(weight, WeightSchedule) and quantity_weights is not None:
            # Merge quantity_weights into existing WeightSchedule
            weight.component_weights = {**weight.component_weights, **quantity_weights}
        
        super().__init__(
            weight=weight,
            name=name,
            data_dim=data_dim,
            field_names=field_names,
            norm_helper=norm_helper,
        )

        self.conserved_keys: Tuple[str, ...] = tuple(conserved_keys or ())
        self.boundary_keys: Tuple[str, ...] = tuple(boundary_keys or ())
        self.use_boundary_fluxes = use_boundary_fluxes
        self.eps = eps
        self.last_components: Dict[str, torch.Tensor] = {}
        self.normalization = normalization
        self._component_names = component_names
        self._boundary_patch_whitelist = boundary_patch_whitelist

        # Build registries
        self._domain_quantity_registry: Dict[str, DomainQuantity] = (
            self._build_domain_quantity_registry()
        )
        self._boundary_flux_registry: Dict[str, BoundaryFluxQuantity] = (
            self._build_boundary_flux_registry()
        )

        # Resolve quantities
        self.domain_quantities: List[DomainQuantity] = []
        for key in self.conserved_keys:
            if key not in self._domain_quantity_registry:
                raise KeyError(
                    f"Unknown conserved key '{key}' for cRMSELoss. "
                    f"Available: {list(self._domain_quantity_registry.keys())}"
                )
            self.domain_quantities.append(self._domain_quantity_registry[key])

        self.boundary_flux_quantities: List[BoundaryFluxQuantity] = []
        if self.use_boundary_fluxes:
            for key in self.boundary_keys:
                if key not in self._boundary_flux_registry:
                    raise KeyError(
                        f"Unknown boundary key '{key}' for cRMSELoss. "
                        f"Available: {list(self._boundary_flux_registry.keys())}"
                    )
                self.boundary_flux_quantities.append(self._boundary_flux_registry[key])

    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        input_frames: Optional[torch.Tensor],
        return_detailed: bool = False,
        keep_bc_dims: bool = False,
        preserve_component_grads: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Denormalize fields
        pred_fields = self.norm_helper.denormalize_to_fields(predictions)
        true_fields = self.norm_helper.denormalize_to_fields(labels)

        # Compute domain-integrated quantities
        domain_pred = self._compute_domain_conserved(pred_fields)
        domain_true = self._compute_domain_conserved(true_fields)

        # Compute normalized errors for domain quantities (raw differences)
        domain_loss_dict = self._compute_cRMSE_dict(
            domain_pred,
            domain_true,
            prefix="domain",
            keep_bc_dims=keep_bc_dims,
        )

        # Compute boundary fluxes if enabled
        boundary_loss_dict: Dict[str, torch.Tensor] = {}
        if self.use_boundary_fluxes and len(self.boundary_keys) > 0:
            if any(k not in domain_true for k in self.boundary_keys):
                extra_domain = self._compute_domain_conserved_for_keys(
                    true_fields,
                    [k for k in self.boundary_keys if k not in domain_true],
                )
                if extra_domain:
                    domain_true = {**domain_true, **extra_domain}

            flux_pred = self._compute_boundary_fluxes(pred_fields)
            flux_true = self._compute_boundary_fluxes(true_fields)
            boundary_loss_dict = self._compute_cRMSE_boundary_dict(
                flux_pred,
                flux_true,
                domain_true,
                keep_bc_dims=keep_bc_dims,
            )

        # All components contain raw normalized differences
        all_components = {**domain_loss_dict, **boundary_loss_dict}

        component_weight_keys = list(self.weight_schedule.component_weights.keys())
        if component_weight_keys:
            missing = [k for k in component_weight_keys if k not in all_components]
            extra = [k for k in all_components.keys() if k not in self.weight_schedule.component_weights]
            if missing or extra:
                raise ValueError(
                    "IntegralConservationRMSE: component weights must match outputs. "
                    f"Missing: {missing}, Extra: {extra}"
                )
        
        # Store detailed components (raw differences)
        self.last_components = {k: v.detach() for k, v in all_components.items()}

        # Keep batch and component dims (for rollout metrics)
        if keep_bc_dims:
            if component_weight_keys:
                component_order = component_weight_keys
            elif self._component_names is not None:
                component_order = list(self._component_names)
            else:
                component_order = sorted(all_components.keys())

            per_component = []
            for name in component_order:
                value = all_components[name]
                if value.ndim == 0:
                    value = value.expand(predictions.shape[0])
                elif value.ndim != 1 or value.shape[0] != predictions.shape[0]:
                    raise ValueError(
                        f"IntegralConservationRMSE: component '{name}' has incompatible shape {tuple(value.shape)}."
                    )

                q_weight = self.weight_schedule.get_loss_component_weight(name)
                q_weight_tensor = torch.tensor(
                    q_weight,
                    device=value.device,
                    dtype=value.dtype,
                )
                weighted_value = self.weight_schedule.base_weight * torch.sqrt(q_weight_tensor) * value
                per_component.append(weighted_value)

            per_component_tensor = torch.stack(per_component, dim=1)

            return per_component_tensor
        else:
            total_squared = torch.zeros((), device=predictions.device, dtype=predictions.dtype)

        for name, value in all_components.items():
            q_weight = self.weight_schedule.get_loss_component_weight(name)
            total_squared = total_squared + q_weight * (value ** 2)

        # Take square root to get RMSE (per-sample if keep_bc_dims)
        total = torch.sqrt(total_squared + self.eps)
        
        # Apply base weight
        weighted_total = self.weight_schedule.base_weight * total

        if not return_detailed:
            return weighted_total

        # Build detailed breakdown with weighted components
        detailed_components: Dict[str, torch.Tensor] = {}
        for name, value in all_components.items():
            q_weight = self.weight_schedule.get_loss_component_weight(name)
            q_weight_tensor = torch.tensor(
                q_weight,
                device=value.device,
                dtype=value.dtype,
            )
            weighted_value = self.weight_schedule.base_weight * torch.sqrt(q_weight_tensor) * value
            detailed_components[name] = weighted_value if preserve_component_grads else weighted_value.detach()

        detailed = {
            "per_component": detailed_components
        }

        return weighted_total, detailed

    def _build_domain_quantity_registry(self) -> Dict[str, DomainQuantity]:
        """
        Map conserved quantity keys to DomainQuantity implementations.

        Available quantities:
        - mass: ∫ ρ dV
        - Px, Py, Pz: ∫ ρ u_i dV
        - kinetic_energy: ∫ 0.5 ρ |u|² dV
        - energy: ∫ E dV
        - enstrophy: ∫ 0.5 |ω|² dV
        - divergence: ∫ |∇·u|² dV
        - center_of_gravity_x/y/z: ∫ x_i ρ dV / ∫ ρ dV
        """
        registry: Dict[str, DomainQuantity] = {}

        registry["mass"] = TotalMass(density_key="Density")
        registry["Px"] = MomentumComponent(direction="x")
        registry["Py"] = MomentumComponent(direction="y")
        registry["Pz"] = MomentumComponent(direction="z")
        registry["kinetic_energy"] = KineticEnergy()

        axis_labels = _axis_labels_from_dim(self.data_dim or 2)
        for axis_label in axis_labels:
            key = f"center_of_gravity_{axis_label}"
            registry[key] = CenterOfGravityAxis(
                axis_label=axis_label,
                density_key="Density",
                name=key,
            )

        registry["energy"] = TotalEnergy(
        energy_key="Energy",
        name="energy",
        )

        registry["enstrophy"] = Enstrophy(
            vort_key="Vorticity",
            name="enstrophy",
        )

        # Grid spacings for divergence computation
        spacings = None
        if hasattr(self, "dx") and hasattr(self, "dy"):
            if hasattr(self, "dz"):
                spacings = [self.dx, self.dy, self.dz]
            else:
                spacings = [self.dx, self.dy]

        registry["divergence"] = DivergenceMeasure(
            vel_keys=("Velocity_X", "Velocity_Y"),
            spacings=spacings,
            name="divergence",
        )

        return registry

    def _build_boundary_flux_registry(self) -> Dict[str, BoundaryFluxQuantity]:
        """
        Map boundary flux keys to BoundaryFluxQuantity implementations.

        Available fluxes:
        - mass: ∫_Γ ρ (u·n) dS
        - Px, Py, Pz: ∫_Γ [ρ u_i (u·n) + p n_i] dS
        - energy: ∫_Γ (E + p)(u·n) dS
        """
        registry: Dict[str, BoundaryFluxQuantity] = {}

        vel_keys = ("Velocity_X", "Velocity_Y", "Velocity_Z")[:self.data_dim or 2]
        pressure_key = "Pressure" if (self.field_names is None or "Pressure" in self.field_names) else None

        registry["mass"] = MassFlux(
            density_key="Density",
            vel_keys=vel_keys,
            name="mass",
        )
        registry["Px"] = MomentumFluxComponent(
            direction="x",
            density_key="Density",
            pressure_key=pressure_key,
            vel_keys=vel_keys,
        )
        registry["Py"] = MomentumFluxComponent(
            direction="y",
            density_key="Density",
            pressure_key=pressure_key,
            vel_keys=vel_keys,
        )
        registry["Pz"] = MomentumFluxComponent(
            direction="z",
            density_key="Density",
            pressure_key=pressure_key,
            vel_keys=vel_keys,
        )
        registry["energy"] = EnergyFlux(
            energy_key="Energy",
            pressure_key="Pressure",
            vel_keys=vel_keys,
            name="energy",
        )

        return registry
    
    def _compute_domain_conserved(
        self,
        fields: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute domain-integrated quantities.

        Parameters
        ----------
        fields : dict of str -> Tensor
            Each tensor has shape (B, T, *spatial).

        Returns
        -------
        dict of str -> Tensor
            Maps quantity_name to time series of shape (B, T).
        """
        sample = next(iter(fields.values()))
        dv = torch.as_tensor(
            getattr(self, "cell_volume", 1.0),
            dtype=sample.dtype,
            device=sample.device,
        )

        out: Dict[str, torch.Tensor] = {}
        for q in self.domain_quantities:
            if not all(f in fields for f in q.required_fields):
                missing = [f for f in q.required_fields if f not in fields]
                raise ValueError(
                    f"IntegralConservationRMSE: missing fields for '{q.name}': {missing}"
                )

            series = q(fields, dv)  # (B, T)
            out[q.name] = series

        return out

    def _compute_domain_conserved_for_keys(
        self,
        fields: Dict[str, torch.Tensor],
        keys: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute domain-integrated quantities for specific keys.

        Parameters
        ----------
        fields : dict of str -> Tensor
            Each tensor has shape (B, T, *spatial).
        keys : sequence of str
            Quantity keys to compute.

        Returns
        -------
        dict of str -> Tensor
            Maps quantity_name to time series of shape (B, T).
        """
        if not keys:
            return {}

        sample = next(iter(fields.values()))
        dv = torch.as_tensor(
            getattr(self, "cell_volume", 1.0),
            dtype=sample.dtype,
            device=sample.device,
        )

        out: Dict[str, torch.Tensor] = {}
        for key in keys:
            if key not in self._domain_quantity_registry:
                continue

            q = self._domain_quantity_registry[key]
            if not all(f in fields for f in q.required_fields):
                missing = [f for f in q.required_fields if f not in fields]
                raise ValueError(
                    f"IntegralConservationRMSE: missing fields for '{q.name}': {missing}"
                )

            series = q(fields, dv)  # (B, T)
            out[q.name] = series

        return out

    def _compute_boundary_fluxes(
        self,
        fields: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute boundary fluxes.

        Returns
        -------
        dict of str -> dict of str -> Tensor
            Nested dict: quantity_key -> patch_name -> flux series (B, T).
        """
        if not self.use_boundary_fluxes or not self.boundary_flux_quantities:
            return {}

        sample = next(iter(fields.values()))  # (B, T, *spatial)
        n_spatial = sample.ndim - 2

        # Get grid spacings
        if hasattr(self, "dx") and hasattr(self, "dy"):
            if n_spatial == 2:
                spacings = [self.dx, self.dy]
            elif n_spatial == 3 and hasattr(self, "dz"):
                spacings = [self.dx, self.dy, self.dz]
            else:
                spacings = [1.0] * n_spatial
        else:
            spacings = [1.0] * n_spatial

        # Define boundary patches
        axis_labels = _axis_labels_from_dim(n_spatial)
        patches: List[BoundaryPatch] = []
        for axis, a_name in enumerate(axis_labels):
            patches.append(BoundaryPatch(name=_patch_name(a_name, "min"), axis=axis, side="min", normal_sign=-1.0))
            patches.append(BoundaryPatch(name=_patch_name(a_name, "max"), axis=axis, side="max", normal_sign=+1.0))

        out: Dict[str, Dict[str, torch.Tensor]] = {}

        for flux_quantity in self.boundary_flux_quantities:
            if not all(f in fields for f in flux_quantity.required_fields):
                missing = [f for f in flux_quantity.required_fields if f not in fields]
                raise ValueError(
                    f"IntegralConservationRMSE: missing fields for '{flux_quantity.name}': {missing}"
                )

            flux_patches = patches
            if self._boundary_patch_whitelist is not None:
                allowed = self._boundary_patch_whitelist.get(flux_quantity.name, ())
                if allowed:
                    allowed_set = set(allowed)
                    flux_patches = [p for p in patches if p.name in allowed_set]

            patch_fluxes = flux_quantity(fields, flux_patches, spacings)
            out[flux_quantity.name] = patch_fluxes

        return out

    def _compute_cRMSE_dict(
        self,
        series_pred: Dict[str, torch.Tensor],
        series_true: Dict[str, torch.Tensor],
        prefix: str = "",
        keep_bc_dims: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute label-normalized raw difference for each quantity.

        error_normalized = (pred - true) / (true + eps)

        Parameters
        ----------
        series_pred : dict of str -> Tensor
            Predicted series (B, T, ...).
        series_true : dict of str -> Tensor
            True series (B, T, ...).
        prefix : str
            Prefix for loss component names.

        Returns
        -------
        dict of str -> Tensor
            Maps f"{prefix}/{key}" to scalar dimensionless difference.
        """
        if not series_true:
            return {}

        loss_dict: Dict[str, torch.Tensor] = {}
        
        for key in self.conserved_keys:
            if key not in series_pred or key not in series_true:
                continue

            pred = series_pred[key]
            true = series_true[key]

            scaled = self._mean_normalized_diff(pred, true, keep_bc_dims=keep_bc_dims)

            name = f"{prefix}/{key}" if prefix else key
            loss_dict[name] = scaled

        return loss_dict

    def _compute_cRMSE_boundary_dict(
        self,
        flux_pred: Dict[str, Dict[str, torch.Tensor]],
        flux_true: Dict[str, Dict[str, torch.Tensor]],
        domain_true: Dict[str, torch.Tensor],
        keep_bc_dims: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute label-normalized raw difference for boundary fluxes.

        Parameters
        ----------
        flux_pred : dict of str -> dict of str -> Tensor
            Nested: quantity -> patch -> predicted flux (B, T).
        flux_true : dict of str -> dict of str -> Tensor
            Nested: quantity -> patch -> true flux (B, T).

        Returns
        -------
        dict of str -> Tensor
            Maps "boundary/{quantity}/{patch}" to scalar dimensionless difference.
        """
        loss_dict: Dict[str, torch.Tensor] = {}

        for q_key in self.boundary_keys:
            if q_key not in flux_pred or q_key not in flux_true:
                continue

            domain_series = domain_true.get(q_key)

            for patch_name, true_series in flux_true[q_key].items():
                if patch_name not in flux_pred[q_key]:
                    continue

                pred_series = flux_pred[q_key][patch_name]

                scaled = self._mean_normalized_diff_with_denom(
                    pred_series,
                    true_series,
                    denom_series=domain_series,
                    keep_bc_dims=keep_bc_dims,
                )

                name = f"boundary/{q_key}/{patch_name}"
                loss_dict[name] = scaled

        return loss_dict

    @staticmethod
    def _mean_diff(
        pred: torch.Tensor,
        true: torch.Tensor,
        keep_bc_dims: bool = False,
    ) -> torch.Tensor:
        """Compute mean absolute difference over all non-batch dimensions."""
        diff = pred - true
        if keep_bc_dims:
            reduce_dims = list(range(1, diff.ndim))
        else:
            reduce_dims = list(range(0, diff.ndim))
        return torch.mean(torch.abs(diff), dim=reduce_dims)

    def _mean_normalized_diff(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        keep_bc_dims: bool = False,
    ) -> torch.Tensor:
        """Compute mean absolute relative difference over all non-batch dims."""
        diff = pred - true
        denom = torch.abs(true)
        rel = torch.abs(diff) / (denom + self.eps)
        if keep_bc_dims:
            reduce_dims = list(range(1, rel.ndim))
        else:
            reduce_dims = list(range(0, rel.ndim))
        return torch.mean(rel, dim=reduce_dims)

    def _mean_normalized_diff_with_denom(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        denom_series: Optional[torch.Tensor] = None,
        keep_bc_dims: bool = False,
    ) -> torch.Tensor:
        """Compute mean absolute relative difference using an external denominator series."""
        diff = pred - true
        if denom_series is None:
            denom = torch.abs(true)
        else:
            denom = torch.abs(denom_series)

        rel = torch.abs(diff) / (denom + self.eps)
        if keep_bc_dims:
            reduce_dims = list(range(1, rel.ndim))
        else:
            reduce_dims = list(range(0, rel.ndim))
        return torch.mean(rel, dim=reduce_dims)


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

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


def _patch_name(axis_label: str, side: str) -> str:
    """
    Convert axis + side to compass/top-bottom names.
    x: west/east, y: south/north, z: bottom/top.
    """
    side_map = {
        "x": {"min": "west", "max": "east"},
        "y": {"min": "south", "max": "north"},
        "z": {"min": "bottom", "max": "top"},
    }
    if axis_label not in side_map or side not in side_map[axis_label]:
        raise ValueError(f"Invalid axis/side: {axis_label}/{side}")
    return side_map[axis_label][side]


def _vel_key_for_axis(vel_keys: Sequence[str], axis_label: str) -> str:
    """
    Map axis label to the corresponding velocity key.
    vel_keys is in (x, y, z) order.
    """
    label_order = ("x", "y", "z")
    key_map = {label: vel_keys[i] for i, label in enumerate(label_order[: len(vel_keys)])}
    if axis_label not in key_map:
        raise ValueError(
            f"Velocity key for axis '{axis_label}' not available in {list(vel_keys)}."
        )
    return key_map[axis_label]

def integrate_over_domain(field: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
    """
    Integrate field over domain: Q = ∫ f dV

    Parameters
    ----------
    field : Tensor of shape (B, T, *spatial)
    dv : scalar Tensor (cell volume)

    Returns
    -------
    Tensor of shape (B, T)
    """
    spatial_dims = tuple(range(2, field.ndim))
    return (field * dv).sum(dim=spatial_dims)

def integrate_over_face(face_field: torch.Tensor, area_per_cell: float) -> torch.Tensor:
    """
    Integrate over boundary face: Φ = ∫ f dS

    Parameters
    ----------
    face_field : Tensor of shape (B, T, *face_spatial)
    area_per_cell : float (cell face area)

    Returns
    -------
    Tensor of shape (B, T)
    """
    area = torch.as_tensor(
        area_per_cell,
        dtype=face_field.dtype,
        device=face_field.device,
    )
    spatial_dims = tuple(range(2, face_field.ndim))
    return (face_field * area).sum(dim=spatial_dims)


# ------------------------------------------------------------------
# Domain quantity implementations
# ------------------------------------------------------------------

class TotalMass(DomainQuantity):
    """Domain-integrated mass: M = ∫ ρ dV"""

    def __init__(self, density_key: str = "Density"):
        super().__init__(name="mass", required_fields=[density_key])
        self.density_key = density_key

    def __call__(
        self,
        fields: Dict[str, torch.Tensor],
        dv: torch.Tensor,
    ) -> torch.Tensor:
        rho = fields[self.density_key]
        return integrate_over_domain(rho, dv)


class CenterOfGravityAxis(DomainQuantity):
    """Center of gravity along a single axis from density: x̄_i = ∫ x_i ρ dV / ∫ ρ dV"""

    def __init__(
        self,
        axis_label: str,
        density_key: str = "Density",
        name: Optional[str] = None,
        eps: float = 1e-12,
    ):
        if axis_label not in ("x", "y", "z"):
            raise ValueError(f"CenterOfGravityAxis: invalid axis '{axis_label}'.")
        axis_name = name or f"center_of_gravity_{axis_label}"
        super().__init__(name=axis_name, required_fields=[density_key])
        self.axis_label = axis_label
        self.density_key = density_key
        self.eps = eps

    def __call__(
        self,
        fields: Dict[str, torch.Tensor],
        dv: torch.Tensor,
    ) -> torch.Tensor:
        rho = fields[self.density_key]
        n_spatial = rho.ndim - 2
        axis_labels = _axis_labels_from_dim(n_spatial)
        if self.axis_label not in axis_labels:
            raise ValueError(
                f"CenterOfGravityAxis: axis '{self.axis_label}' not available for "
                f"n_spatial={n_spatial}."
            )

        axis_index = axis_labels.index(self.axis_label)
        dim = 2 + axis_index
        size = rho.shape[dim]

        coords = torch.arange(size, device=rho.device, dtype=rho.dtype)

        shape = [1, 1] + [1] * n_spatial
        shape[2 + axis_index] = size
        coord_grid = coords.view(*shape)

        weighted = rho * coord_grid
        numerator = integrate_over_domain(weighted, dv)
        denominator = integrate_over_domain(rho, dv)
        return numerator / (denominator + self.eps)


class MomentumComponent(DomainQuantity):
    """Domain-integrated momentum component: P_i = ∫ ρ u_i dV"""

    def __init__(
        self,
        direction: str,
        density_key: str = "Density",
        vel_key: Optional[str] = None,
    ):
        assert direction in ("x", "y", "z")
        if vel_key is None:
            vel_key = {
                "x": "Velocity_X",
                "y": "Velocity_Y",
                "z": "Velocity_Z",
            }[direction]

        name = f"P{direction}"
        super().__init__(name=name, required_fields=[density_key, vel_key])
        self.density_key = density_key
        self.vel_key = vel_key

    def __call__(
        self,
        fields: Dict[str, torch.Tensor],
        dv: torch.Tensor,
    ) -> torch.Tensor:
        rho = fields[self.density_key]
        u = fields[self.vel_key]
        mom_density = rho * u
        return integrate_over_domain(mom_density, dv)


class KineticEnergy(DomainQuantity):
    """Domain-integrated kinetic energy: KE = ∫ 0.5 ρ |u|² dV"""

    def __init__(
        self,
        density_key: str = "Density",
        vel_keys: Sequence[str] = ("Velocity_X", "Velocity_Y"),
        name: str = "kinetic_energy",
    ):
        super().__init__(name=name, required_fields=[density_key, *vel_keys])
        self.density_key = density_key
        self.vel_keys = tuple(vel_keys)

    def __call__(
        self,
        fields: Dict[str, torch.Tensor],
        dv: torch.Tensor,
    ) -> torch.Tensor:
        rho = fields[self.density_key]
        speed_sq = 0.0
        for vk in self.vel_keys:
            v = fields[vk]
            speed_sq = speed_sq + v**2 

        K_density = 0.5 * rho * speed_sq
        return integrate_over_domain(K_density, dv)


class Enstrophy(DomainQuantity):
    """Domain-integrated enstrophy: Ω = ∫ 0.5 |ω|² dV"""

    def __init__(self, vort_key: str = "Vorticity", name: str = "enstrophy"):
        super().__init__(name=name, required_fields=[vort_key])
        self.vort_key = vort_key

    def __call__(self, fields: Dict[str, torch.Tensor], dv: torch.Tensor) -> torch.Tensor:
        omega = fields[self.vort_key]
        enstrophy_density = 0.5 * omega**2
        return integrate_over_domain(enstrophy_density, dv)


class TotalEnergy(DomainQuantity):
    """
    Domain-integrated total energy: E_tot = ∫ E dV
    """

    def __init__(
        self,
        energy_key: str = "Energy",
        name: str = "energy",
    ):
        super().__init__(name=name, required_fields=[energy_key])
        self.energy_key = energy_key

    def __call__(self, fields: Dict[str, torch.Tensor], dv: torch.Tensor) -> torch.Tensor:
        E = fields[self.energy_key]
        return integrate_over_domain(E, dv)


class DivergenceMeasure(DomainQuantity):
    """
    Domain-integrated squared divergence: D = ∫ |∇·u|² dV

    Uses central finite differences with periodic boundaries via torch.roll.
    """

    def __init__(
        self,
        vel_keys: Sequence[str] = ("Velocity_X", "Velocity_Y"),
        spacings: Optional[Sequence[float]] = None,
        name: str = "divergence",
    ):
        super().__init__(name=name, required_fields=list(vel_keys))
        self.vel_keys = tuple(vel_keys)
        self.spacings = spacings

    def __call__(self, fields: Dict[str, torch.Tensor], dv: torch.Tensor) -> torch.Tensor:
        u0 = fields[self.vel_keys[0]]
        ndim = u0.ndim
        n_spatial = ndim - 2

        if self.spacings is not None and len(self.spacings) != len(self.vel_keys):
            raise ValueError(
                f"DivergenceMeasure: len(spacings)={len(self.spacings)} "
                f"must match len(vel_keys)={len(self.vel_keys)}."
            )

        spacings = (
            list(self.spacings)
            if self.spacings is not None
            else [1.0] * len(self.vel_keys)
        )

        div = torch.zeros_like(u0)

        for i, (vel_key, dx) in enumerate(zip(self.vel_keys, spacings)):
            u = fields[vel_key]
            dim = 2 + i  # spatial axis: x=2, y=3, z=4

            # Central difference with periodic roll
            u_plus = torch.roll(u, shifts=-1, dims=dim)
            u_minus = torch.roll(u, shifts=1, dims=dim)
            du_dx = (u_plus - u_minus) / (2.0 * dx)
            div = div + du_dx

        div_sq = div**2
        return integrate_over_domain(div_sq, dv)


# ------------------------------------------------------------------
# Boundary flux implementations
# ------------------------------------------------------------------

class MassFlux(BoundaryFluxQuantity):
    """Boundary mass flux: Φ_M = ∫_Γ ρ (u·n) dS"""

    def __init__(
        self,
        density_key: str = "Density",
        vel_keys: Sequence[str] = ("Velocity_X", "Velocity_Y", "Velocity_Z"),
        name: str = "mass",
    ):
        super().__init__(name=name, required_fields=[density_key, *vel_keys])
        self.density_key = density_key
        self.vel_keys = tuple(vel_keys)

    def __call__(
        self,
        fields: Dict[str, torch.Tensor],
        patches: Sequence[BoundaryPatch],
        spacings: Sequence[float],
    ) -> Dict[str, torch.Tensor]:
        rho = fields[self.density_key]
        n_spatial = rho.ndim - 2
        vel_keys = self.vel_keys[:n_spatial]
        axis_labels = _axis_labels_from_dim(n_spatial)

        results: Dict[str, torch.Tensor] = {}

        for patch in patches:
            axis = patch.axis
            dim = 2 + axis
            axis_label = axis_labels[axis]

            # u·n = n_sign * u_axis  (normal aligned with axis)
            u_axis = fields[_vel_key_for_axis(vel_keys, axis_label)]
            un = patch.normal_sign * u_axis

            flux_density = rho * un

            # Extract boundary face
            sl = [slice(None)] * flux_density.ndim
            sl[dim] = 0 if patch.side == "min" else -1
            face_field = flux_density[tuple(sl)]

            # Compute face area
            area = 1.0
            for k, dx in enumerate(spacings):
                if k != axis:
                    area *= dx

            results[patch.name] = integrate_over_face(face_field, area)

        return results


class MomentumFluxComponent(BoundaryFluxQuantity):
    """
    Boundary momentum flux for component i:
    Φ_{P_i} = ∫_Γ [ρ u_i (u·n) + p n_i] dS

    Inviscid approximation (no viscous stresses).
    """

    def __init__(
        self,
        direction: str,
        density_key: str = "Density",
        pressure_key: Optional[str] = "Pressure",
        vel_keys: Sequence[str] = ("Velocity_X", "Velocity_Y", "Velocity_Z"),
    ):
        assert direction in ("x", "y", "z")
        name = f"P{direction}"
        required = [density_key, *vel_keys]
        if pressure_key is not None:
            required.append(pressure_key)
        super().__init__(name=name, required_fields=required)

        self.index = {"x": 0, "y": 1, "z": 2}[direction]
        self.direction_label = direction
        self.density_key = density_key
        self.pressure_key = pressure_key
        self.vel_keys = tuple(vel_keys)

    def __call__(
        self,
        fields: Dict[str, torch.Tensor],
        patches: Sequence[BoundaryPatch],
        spacings: Sequence[float],
    ) -> Dict[str, torch.Tensor]:
        rho = fields[self.density_key]
        p = None
        if self.pressure_key is not None and self.pressure_key in fields:
            p = fields[self.pressure_key]
        n_spatial = rho.ndim - 2

        vel_keys = self.vel_keys[:n_spatial]
        axis_labels = _axis_labels_from_dim(n_spatial)
        results: Dict[str, torch.Tensor] = {}

        for patch in patches:
            axis = patch.axis
            dim = 2 + axis
            axis_label = axis_labels[axis]

            # u·n
            u_axis = fields[_vel_key_for_axis(vel_keys, axis_label)]
            un = patch.normal_sign * u_axis

            # u_i (component being tracked)
            u_i = fields[_vel_key_for_axis(vel_keys, self.direction_label)]

            # n_i (component of normal in direction i)
            n_i = patch.normal_sign if self.direction_label == axis_label else 0.0

            flux_density = rho * u_i * un
            if p is not None:
                flux_density = flux_density + p * n_i

            sl = [slice(None)] * flux_density.ndim
            sl[dim] = 0 if patch.side == "min" else -1
            face_field = flux_density[tuple(sl)]

            area = 1.0
            for k, dx in enumerate(spacings):
                if k != axis:
                    area *= dx

            results[patch.name] = integrate_over_face(face_field, area)

        return results


class EnergyFlux(BoundaryFluxQuantity):
    """Boundary total energy flux: Φ_E = ∫_Γ (E + p)(u·n) dS"""

    def __init__(
        self,
        energy_key: str = "Energy",
        pressure_key: str = "Pressure",
        vel_keys: Sequence[str] = ("Velocity_X", "Velocity_Y", "Velocity_Z"),
        name: str = "energy",
    ):
        required = [energy_key, pressure_key, *vel_keys]
        super().__init__(name=name, required_fields=required)
        self.energy_key = energy_key
        self.pressure_key = pressure_key
        self.vel_keys = tuple(vel_keys)

    def __call__(
        self,
        fields: Dict[str, torch.Tensor],
        patches: Sequence[BoundaryPatch],
        spacings: Sequence[float],
    ) -> Dict[str, torch.Tensor]:
        E = fields[self.energy_key]
        p = fields[self.pressure_key]
        n_spatial = E.ndim - 2

        vel_keys = self.vel_keys[:n_spatial]
        axis_labels = _axis_labels_from_dim(n_spatial)
        results: Dict[str, torch.Tensor] = {}

        for patch in patches:
            axis = patch.axis
            dim = 2 + axis
            axis_label = axis_labels[axis]

            u_axis = fields[_vel_key_for_axis(vel_keys, axis_label)]
            un = patch.normal_sign * u_axis

            flux_density = (E + p) * un

            sl = [slice(None)] * flux_density.ndim
            sl[dim] = 0 if patch.side == "min" else -1
            face_field = flux_density[tuple(sl)]

            area = 1.0
            for k, dx in enumerate(spacings):
                if k != axis:
                    area *= dx

            results[patch.name] = integrate_over_face(face_field, area)

        return results