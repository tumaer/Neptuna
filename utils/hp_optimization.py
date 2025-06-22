from __future__ import annotations

"""Utility helpers for performing hyper-parameter optimisation with HuggingFace Trainer.

This module encapsulates all logic that is required by the main training
pipeline to run Optuna-based hyper-parameter optimisation.  Placing it in a
stand-alone file keeps `main.py` focused on the high-level training flow while
still allowing easy reuse of these helpers elsewhere.
"""

from typing import Dict, List
from collections.abc import Mapping
import copy
import json
from typing import Union
from omegaconf import ListConfig

# ---------------------------------------------------------------------------
# Simple (de)serialization helpers originally in utils.hp_codec
# ---------------------------------------------------------------------------

# NOTE: Kept here to avoid an extra module.

def encode(value):  # noqa: D401
    """Convert list/ListConfig to a JSON string for safe hashing by Optuna.

    Any other type is returned unchanged.
    """
    if isinstance(value, (list, ListConfig)):
        # Ensure plain list before dumping.
        # Only lists are encoded into strings (requirement by Optuna for categorical HPs)
        return json.dumps(list(value))
    return value


def decode(value):  # noqa: D401
    """Attempt to convert an encoded JSON string back to ListConfig.

    If *value* was not produced by :pyfunc:`encode`, it is returned unchanged.
    """
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value  # Not JSON – definitely not encoded by encode().

    # Only lists are encoded into strings (requirement by Optuna for categorical HPs); leave other JSON literals untouched.
    if isinstance(parsed, list):
        return ListConfig(parsed)
    return value


def compute_objective_function(selected_metrics: List[str]):
    """Return a callable that aggregates evaluation metrics into a single score.

    The returned function has the signature expected by
    `transformers.Trainer.hyperparameter_search` (`outputs` tuple) and returns a
    *scalar* objective value that Optuna will attempt to **minimise**.  All
    speed-related metrics are ignored.  If *selected_metrics* is empty, the
    function falls back to *eval_loss* (if available) or zero.
    """

    def compute_objective(outputs: Union[tuple[Dict[str, float]], Dict[str, float]]): 
        
        #it is a tuple when not even a single evaluation was done during training (like when max_steps < eval_steps), 
        # then the outputs is a tuple containing metrics, predictions, labels and inputs.
        if isinstance(outputs, tuple):
            metrics = copy.deepcopy(outputs[0])
        else:
            metrics = copy.deepcopy(outputs)
        
        loss = metrics.pop("eval_loss", None)
        _ = metrics.pop("epoch", None)

        # Drop speed metrics (e.g. *_runtime, *_per_second)
        speed_metrics = [m for m in metrics if m.endswith("_runtime") or m.endswith("_per_second")]
        for sm in speed_metrics:
            metrics.pop(sm, None)

        # Aggregate the requested metrics.
        selected = [metrics[m] for m in selected_metrics if m in metrics]
        return sum(selected) if selected else (loss if loss is not None else 0.0)

    return compute_objective


# ---------------------------------------------------------------------------
# Optuna search-space helpers
# ---------------------------------------------------------------------------

def optuna_hp_space_factory(config):
    """Return a *factory* that builds an Optuna search-space callable.

    The callable returned by this factory has the signature expected by
    `transformers.Trainer.hyperparameter_search` (i.e. it takes a *trial* and
    returns a `dict[str, Any]` mapping of fully-qualified config paths to
    suggested values).
    """

    search_space = config["hyperparam_opt_config"]["search_space"]

    def suggest_from_spec(param_path: str, spec: Mapping, trial):
        """Suggest a value for a *trial* based on the given *spec* definition."""
        method = spec["method"]
        if method == "float":
            return trial.suggest_float(param_path, spec["low"], spec["high"], log=spec.get("log", False))
        if method == "int":
            return trial.suggest_int(param_path, spec["low"], spec["high"], log=spec.get("log", False))
        if method == "categorical":
            choices = spec["choices"]
            if isinstance(choices, ListConfig):
                choices = list(choices)
            processed_choices = [encode(c) for c in choices]
            sampled = trial.suggest_categorical(param_path, processed_choices)
            return decode(sampled)
        raise ValueError(f"Unsupported suggestion method: {method}")

    def traverse(node: Mapping, path_parts: List[str], trial):
        suggestions: Dict[str, object] = {}
        if not isinstance(node, Mapping):
            return suggestions

        if "method" in node:  # leaf spec
            dot_path = ".".join(path_parts)
            suggestions[dot_path] = suggest_from_spec(dot_path, node, trial)
            return suggestions

        # Recurse into children
        for key, child in node.items():
            suggestions.update(traverse(child, path_parts + [key], trial))
        return suggestions

    def optuna_hp_space(trial):  # noqa: D401
        """Optuna *hp_space* callback compatible with HuggingFace Trainer."""
        return traverse(search_space, [], trial)

    return optuna_hp_space 