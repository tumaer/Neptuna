from __future__ import annotations
from transformers.hyperparameter_search import OptunaBackend as OptunaBackend_
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS
from transformers.trainer_utils import HPSearchBackend

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
import torch
import os
from transformers.trainer_utils import BestRun, PREFIX_CHECKPOINT_DIR
from transformers.training_args import ParallelMode


def encode(value):  
    """Convert list/ListConfig to a JSON string for safe hashing by Optuna.

    Any other type is returned unchanged.
    """
    if isinstance(value, (list, ListConfig)):
        # Ensure plain list before dumping.
        # Only lists are encoded into strings (requirement by Optuna for categorical HPs)
        return json.dumps(list(value))
    return value


def decode(value):  
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

    search_domain = config["hyperparam_opt_config"]["search_domain"]

    def suggest_from_spec(param_path: str, spec: Mapping, trial):
        """Suggest a value for a *trial* based on the given *spec* definition."""
        method = spec["method"]
        if method == "float":
            if spec.get("log", False):
                return trial.suggest_float(param_path, spec["low"], spec["high"], log=True)
            else:
                return trial.suggest_float(param_path, spec["low"], spec["high"], step=spec.get("step", 1), log=False)
        if method == "int":
            if spec.get("log", False):
                return trial.suggest_int(param_path, spec["low"], spec["high"], log=True)
            else:
                return trial.suggest_int(param_path, spec["low"], spec["high"], step=spec.get("step", 1), log=False)
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

    def optuna_hp_space(trial): 
        """Optuna *hp_space* callback compatible with HuggingFace Trainer."""
        return traverse(search_domain, [], trial)

    return optuna_hp_space 


# ---------------------------------------------------------------------------
# Optuna sampler selection
# ---------------------------------------------------------------------------

def get_optuna_sampler(sampler_name: str, config=None, **kwargs):
    """Create an Optuna sampler based on the given name and optional parameters.
    
    Args:
        sampler_name: Name of the sampler to create. Supported options:
            - "TPE_sampler" or "TPESampler": Tree-structured Parzen Estimator
            - "Random_sampler" or "RandomSampler": Random sampling  
            - "Grid_sampler" or "GridSampler": Grid search
            - "CMA_ES_sampler" or "CmaEsSampler": CMA-ES algorithm
            - "QMC_sampler" or "QMCSampler": Quasi-Monte Carlo sampling
            - "NSGA2_sampler" or "NSGAIISampler": Multi-objective optimization
            - "Partial_sampler" or "PartialFixedSampler": Fix some parameters
        config: Optional config dict containing 'sampler_params' section
        **kwargs: Additional parameters to pass to the sampler constructor
        
    Returns:
        An instance of the requested Optuna sampler
        
    Raises:
        ValueError: If the sampler name is not supported
        ImportError: If required dependencies are not available
    """
    
    # Extract sampler parameters from config if provided
    sampler_params = {}
    if config and "hyperparam_opt_config" in config and "sampler_params" in config["hyperparam_opt_config"]:
        sampler_params = dict(config["hyperparam_opt_config"]["sampler_params"])
    
    # Merge with explicit kwargs (kwargs take precedence)
    sampler_params.update(kwargs)
    
    # Normalize sampler name (handle both snake_case and CamelCase)
    sampler_name = sampler_name.lower()
    
    try:
        if sampler_name in ["tpe_sampler", "tpesampler"]:
            from optuna.samplers import TPESampler
            return TPESampler(**sampler_params)
            
        elif sampler_name in ["random_sampler", "randomsampler"]:
            from optuna.samplers import RandomSampler
            return RandomSampler(**sampler_params)
            
        elif sampler_name in ["grid_sampler", "gridsampler"]:
            from optuna.samplers import GridSampler
            if "search_space" not in sampler_params:
                raise ValueError("GridSampler requires 'search_space' argument")
            return GridSampler(**sampler_params)
            
        elif sampler_name in ["cma_es_sampler", "cmaessampler"]:
            from optuna.samplers import CmaEsSampler
            return CmaEsSampler(**sampler_params)
            
        elif sampler_name in ["qmc_sampler", "qmcsampler"]:
            from optuna.samplers import QMCSampler
            return QMCSampler(**sampler_params)
            
        elif sampler_name in ["nsga2_sampler", "nsgaiisampler"]:
            from optuna.samplers import NSGAIISampler
            return NSGAIISampler(**sampler_params)
            
        elif sampler_name in ["partial_sampler", "partialfixedsampler"]:
            from optuna.samplers import PartialFixedSampler
            if "fixed_params" not in sampler_params:
                raise ValueError("PartialFixedSampler requires 'fixed_params' argument")
            return PartialFixedSampler(**sampler_params)
            
        else:
            # List available samplers for better error message
            available = [
                "TPE_sampler", "Random_sampler", "Grid_sampler", 
                "CMA_ES_sampler", "QMC_sampler", "NSGA2_sampler", "Partial_sampler"
            ]
            raise ValueError(
                f"Unsupported sampler: '{sampler_name}'. "
                f"Available options: {', '.join(available)}"
            )
            
    except ImportError as e:
        raise ImportError(
            f"Failed to import {sampler_name}. Make sure optuna is installed "
            f"and all required dependencies are available: {e}"
        ) 


def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import optuna
    from accelerate.utils.memory import release_memory

    if trainer.args.process_index == 0:

        def _objective(trial: optuna.Trial, checkpoint_dir=None):
            checkpoint = None
            if checkpoint_dir:
                for subdir in os.listdir(checkpoint_dir):
                    if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                        checkpoint = os.path.join(checkpoint_dir, subdir)
            trainer.objective = None
            if trainer.args.world_size > 1:
                if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                    raise RuntimeError("only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.")
                trainer.hp_space(trial)
                fixed_trial = optuna.trial.FixedTrial(trial.params, trial.number)
                trial_main_rank_list = [fixed_trial]
                torch.distributed.broadcast_object_list(trial_main_rank_list, src=0)
                trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
            else:
                trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
                print('\n')
        
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)

            # Free GPU memory
            trainer.model_wrapped, trainer.model = release_memory(trainer.model_wrapped, trainer.model)
            trainer.accelerator.clear()

            return trainer.objective

        timeout = kwargs.pop("timeout", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        gc_after_trial = kwargs.pop("gc_after_trial", False)
        directions = direction if isinstance(direction, list) else None
        direction = None if directions is not None else direction
        study = optuna.create_study(direction=direction, directions=directions, **kwargs)
        #NOTE: catch=(RuntimeError,) is added on top of the default function in transformers.integrations.integration_utils.py
        study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, gc_after_trial=gc_after_trial, catch=(RuntimeError,ValueError,))
        if not study._is_multi_objective():
            best_trial = study.best_trial
            return BestRun(str(best_trial.number), best_trial.value, best_trial.params), study
        else:
            best_trials = study.best_trials
            return [BestRun(str(best.number), best.values, best.params) for best in best_trials], study
    else:
        for i in range(n_trials):
            trainer.objective = None
            trial_main_rank_list = [None]
            if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise RuntimeError("only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.")
            torch.distributed.broadcast_object_list(trial_main_rank_list, src=0)
            trainer.train(resume_from_checkpoint=None, trial=trial_main_rank_list[0])
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
        return None


class OptunaBackend(OptunaBackend_):
    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        print("Running OptunaBackend.run")
        return run_hp_search_optuna(trainer, n_trials, direction, **kwargs)
    
ALL_HYPERPARAMETER_SEARCH_BACKENDS[HPSearchBackend(OptunaBackend.name)] = OptunaBackend