from transformers import PreTrainedModel as PreTrainedModel_, PretrainedConfig as PretrainedConfig_
import collections
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import math
import os
import re
import shutil
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
from omegaconf import OmegaConf
from zipfile import is_zipfile

import torch
from torch import Tensor, nn
from huggingface_hub import split_torch_state_dict_into_shards
from packaging import version

from transformers.dynamic_module_utils import custom_object_save

from transformers.loss.loss_utils import LOSS_MAPPING
from transformers.pytorch_utils import (  # noqa: F401
    Conv1D,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    id_tensor_storage,
    prune_conv1d_layer,
    prune_layer,
    prune_linear_layer,
)
from transformers.quantizers import AutoHfQuantizer, HfQuantizer
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.safetensors_conversion import auto_conversion
from transformers.utils import (
    ACCELERATE_MIN_VERSION,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    ModelOutput,
    PushToHubMixin,
    cached_file,
    copy_func,
    download_url,
    extract_commit_hash,
    has_file,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_flash_attn_2_available,
    is_offline_mode,
    is_optimum_available,
    is_peft_available,
    is_remote_url,
    is_safetensors_available,
    is_torch_flex_attn_available,
    is_torch_greater_or_equal,
    is_torch_npu_available,
    is_torch_sdpa_available,
    is_torch_xla_available,
    logging,
    replace_return_docstrings,
    strtobool,
)
from transformers.utils.hub import create_and_tag_model_card, get_checkpoint_shard_files
from transformers.utils.import_utils import (
    ENV_VARS_TRUE_VALUES,
    is_sagemaker_mp_enabled,
    is_torch_fx_proxy,
    is_torchdynamo_compiling,
)
from transformers.utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
from transformers.modeling_utils import (
    get_parameter_dtype, is_sagemaker_mp_enabled, IS_SAGEMAKER_MP_POST_1_10, 
    XLA_USE_BF16, XLA_DOWNCAST_BF16, logger, _add_variant, safe_save_file, get_state_dict_from_offload, accelerate_version,
    _find_identical, _find_disjoint, _get_tied_weight_keys, find_tied_parameters, unwrap_model
)
#import smdistributed.modelparallel.torch as smp


class PreTrainedModel(PreTrainedModel_):

    def __init__(self, config: PretrainedConfig_, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)


    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~PreTrainedModel.from_pretrained`] class method. Taken and adjusted from Transformers PreTrainedModel

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
                without CPU OOM issues.

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            save_peft_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
                keys of the state dict of adapters needs to be pre-pended with `base_model.model`. Advanced users can
                disable this behaviours by setting `save_peft_format` to `False`.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """

        use_auth_token = kwargs.pop("use_auth_token", None)
        ignore_metadata_errors = kwargs.pop("ignore_metadata_errors", False)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        _hf_peft_config_loaded = getattr(self, "_hf_peft_config_loaded", False)

        hf_quantizer = getattr(self, "hf_quantizer", None)
        quantization_serializable = (
            hf_quantizer is not None
            and isinstance(hf_quantizer, HfQuantizer)
            and hf_quantizer.is_serializable(safe_serialization=safe_serialization)
        )

        if hf_quantizer is not None and not _hf_peft_config_loaded and not quantization_serializable:
            raise ValueError(
                f"The model is quantized with {hf_quantizer.quantization_config.quant_method} and is not serializable - check out the warnings from"
                " the logger on the traceback to understand the reason why the quantized model is not serializable."
            )

        if "save_config" in kwargs:
            warnings.warn(
                "`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead."
            )
            is_main_process = kwargs.pop("save_config")
        if safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Unset attn implementation so it can be set to another one when loading back
        model_to_save.config._attn_implementation_autoset = False

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        # Save the config
        if is_main_process:
            if not _hf_peft_config_loaded:
                # If the model config has set attributes that should be in the generation config, move them there.
                misplaced_generation_parameters = model_to_save.config._get_non_default_generation_parameters()
                if self.can_generate() and len(misplaced_generation_parameters) > 0:
                    warnings.warn(
                        "Moving the following attributes in the config to the generation config: "
                        f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                        "generation parameters in the model config, as opposed to in the generation config.",
                        UserWarning,
                    )
                    for param_name, param_value in misplaced_generation_parameters.items():
                        setattr(model_to_save.generation_config, param_name, param_value)
                        setattr(model_to_save.config, param_name, None)
                model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)

            if _hf_peft_config_loaded:
                logger.info(
                    "Detected adapters on the model, saving the model in the PEFT format, only adapter weights will be saved."
                )
                state_dict = model_to_save.get_adapter_state_dict(state_dict=state_dict)

                if save_peft_format:
                    logger.info(
                        "To match the expected format of the PEFT library, all keys of the state dict of adapters will be pre-pended with `base_model.model`."
                    )
                    peft_state_dict = {}
                    for key, value in state_dict.items():
                        peft_state_dict[f"base_model.model.{key}"] = value
                    state_dict = peft_state_dict

                active_adapter = self.active_adapters()

                if len(active_adapter) > 1:
                    raise ValueError(
                        "Multiple active adapters detected, saving multiple active adapters is not supported yet. You can save adapters separately one by one "
                        "by iteratively calling `model.set_adapter(adapter_name)` then `model.save_pretrained(...)`"
                    )
                active_adapter = active_adapter[0]

                current_peft_config = self.peft_config[active_adapter]
                current_peft_config.save_pretrained(save_directory)

        # for offloaded modules
        module_map = {}

        # Save the model
        if state_dict is None:
            # if any model parameters are offloaded, make module map
            if (
                hasattr(self, "hf_device_map")
                and len(set(self.hf_device_map.values())) > 1
                and ("cpu" in self.hf_device_map.values() or "disk" in self.hf_device_map.values())
            ):
                warnings.warn(
                    "Attempting to save a model with offloaded modules. Ensure that unallocated cpu memory exceeds the `shard_size` (5GB default)"
                )
                for name, module in model_to_save.named_modules():
                    if name == "":
                        continue
                    module_state_dict = module.state_dict()

                    for key in module_state_dict:
                        module_map[name + f".{key}"] = module
            state_dict = model_to_save.state_dict()

        # Translate state_dict from smp to hf if saving with smp >= 1.10
        if IS_SAGEMAKER_MP_POST_1_10:
            for smp_to_hf, _ in smp.state.module_manager.translate_functions:
                state_dict = smp_to_hf(state_dict)

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]

        # Rename state_dict keys before saving to file. Do nothing unless overridden in a particular model.
        # (initially introduced with TimmWrapperModel to remove prefix and make checkpoints compatible with timm)
        state_dict = self._fix_state_dict_keys_on_save(state_dict)

        if safe_serialization:
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            for name, tensor in state_dict.items():
                # Sometimes in the state_dict we have non-tensor objects.
                # e.g. in bitsandbytes we have some `str` objects in the state_dict
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    # In the non-tensor case, fall back to the pointer of the object itself
                    ptrs[id(tensor)].append(name)

            # These are all the pointers of shared tensors
            if hasattr(self, "hf_device_map"):
                # if the model has offloaded parameters, we must check using find_tied_parameters()
                tied_params = find_tied_parameters(self)
                if tied_params:
                    tied_names = tied_params[0]
                    shared_ptrs = {
                        ptr: names for ptr, names in ptrs.items() if any(name in tied_names for name in names)
                    }
                else:
                    shared_ptrs = {}
            else:
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

            # Recursively descend to find tied weight keys
            _tied_weights_keys = _get_tied_weight_keys(self)
            error_names = []
            to_delete_names = set()
            for names in shared_ptrs.values():
                # Removing the keys which are declared as known duplicates on
                # load. This allows to make sure the name which is kept is consistent.
                if _tied_weights_keys is not None:
                    found = 0
                    for name in sorted(names):
                        matches_pattern = any(re.search(pat, name) for pat in _tied_weights_keys)
                        if matches_pattern and name in state_dict:
                            found += 1
                            if found < len(names):
                                to_delete_names.add(name)
            # We are entering a place where the weights and the transformers configuration do NOT match.
            shared_names, disjoint_names = _find_disjoint(shared_ptrs.values(), state_dict)
            # Those are actually tensor sharing but disjoint from each other, we can safely clone them
            # Reloaded won't have the same property, but it shouldn't matter in any meaningful way.
            for name in disjoint_names:
                state_dict[name] = state_dict[name].clone()

            # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
            # If the link between tensors was done at runtime then `from_pretrained` will not get
            # the key back leading to random tensor. A proper warning will be shown
            # during reload (if applicable), but since the file is not necessarily compatible with
            # the config, better show a proper warning.
            shared_names, identical_names = _find_identical(shared_names, state_dict)
            # delete tensors that have identical storage
            for inames in identical_names:
                known = inames.intersection(to_delete_names)
                for name in known:
                    del state_dict[name]
                unknown = inames.difference(to_delete_names)
                if len(unknown) > 1:
                    error_names.append(unknown)

            if shared_names:
                error_names.append(set(shared_names))

            if len(error_names) > 0:
                raise RuntimeError(
                    f"The weights trying to be saved contained shared tensors {error_names} that are mismatching the transformers base configuration. Try saving using `safe_serialization=False` or remove this tensor sharing.",
                )

        # Shard the model if it is too big.
        if not _hf_peft_config_loaded:
            weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
            weights_name = _add_variant(weights_name, variant)
        else:
            weights_name = ADAPTER_SAFE_WEIGHTS_NAME if safe_serialization else ADAPTER_WEIGHTS_NAME

        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
        )
        # Save index if sharded
        index = None
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in state_dict_split.filename_to_tensors.keys()
                and is_main_process
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)
        # Save the model
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        if module_map:
            filename_to_tensors = logging.tqdm(filename_to_tensors, desc="Saving checkpoint shards")
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                shard[tensor] = state_dict[tensor].contiguous()
                # delete reference, see https://github.com/huggingface/transformers/pull/34890
                del state_dict[tensor]

            # remake shard with onloaded parameters if necessary
            if module_map:
                if accelerate_version < version.parse("0.31"):
                    raise ImportError(
                        f"You need accelerate version to be greater or equal than 0.31 to save models with offloaded parameters. Detected version {accelerate_version}. "
                        f"Please upgrade accelerate with `pip install -U accelerate`"
                    )
                # init state_dict for this shard
                shard_state_dict = {name: "" for name in shard}
                for module_name in shard:
                    # skip to collect this weight again
                    if shard_state_dict.get(module_name) != "":
                        continue
                    module = module_map[module_name]
                    # update state dict with onloaded parameters
                    shard_state_dict = get_state_dict_from_offload(module, module_name, shard_state_dict)

                # assign shard to be the completed state dict
                shard = shard_state_dict
                del shard_state_dict
                gc.collect()

            if safe_serialization:
                # At some point we will need to deal better with save_function (used for TPU and other distributed
                # joyfulness), but for now this enough.
                safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
            else:
                save_function(shard, os.path.join(save_directory, shard_file))

        del state_dict

        if index is None:
            path_to_weights = os.path.join(save_directory, weights_name)
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

        if push_to_hub:
            # Eventually create an empty model card
            model_card = create_and_tag_model_card(
                repo_id, self.model_tags, token=token, ignore_metadata_errors=ignore_metadata_errors
            )

            # Update model card if needed:
            model_card.save(os.path.join(save_directory, "README.md"))

            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )

class PretrainedConfig(PretrainedConfig_):
    """
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.
     Args:
        in_channels (int): Number of input channels for the model.
        out_channels (int): Number of output channels for the model.
        dimension (int): Dimensionality of the input data (e.g., 1D, 2D, or 3D).
        grid_resolution (Union[int, List[int], Tuple[int]]): Input and output spatial size.
        sequence_info (Optional[List[int]]): Sequence information for the model. Elements of List are: input sequence length, output sequence length, stride
        coord_features (bool): Whether to include coordinate features in the model. Default is True.
        latent_channels (int): Number of latent channels in the model.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        in_channels: int = 1,       # Number of input_channels
        out_channels: int = 1,      # Number of output channels
        dimension: int = 1,
        grid_resolution: Union[int, List[int], Tuple[int]] = [160], # Input and Output spatial size (required )
        sequence_info: Optional[List[int]] = [1,1,1],
        coord_features: bool = True,
        latent_channels: int = 32,
        include_input_seq_len: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension
        self.sequence_info = sequence_info

        if include_input_seq_len:
            self.in_size = self.in_channels * self.sequence_info[0] 
            self.out_size = self.out_channels * self.sequence_info[1]

        self.grid_resolution = grid_resolution
        self.latent_channels = latent_channels

        self.coord_features = coord_features
        # Add relative coordinate feature
        if coord_features:
            self.in_size = self.in_size + dimension

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self._set_token_in_kwargs(kwargs)

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        non_default_generation_parameters = self._get_non_default_generation_parameters()
        if len(non_default_generation_parameters) > 0:
            # TODO (joao): this should be an exception if the user has modified the loaded config. See #33886
            warnings.warn(
                "Some non-default generation parameters are set in the model config. These should go into either a) "
                "`model.generation_config` (as opposed to `model.config`); OR b) a GenerationConfig file "
                "(https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model)."
                "This warning will become an exception in the future."
                f"\nNon-default generation parameters: {str(non_default_generation_parameters)}",
                UserWarning,
            )

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        
        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        # NOTE: default dumpts List objects in the PretrainedConfig to json, since only json.dumps throws an error
        def default(o):
            return OmegaConf.to_container(o, resolve=True)
        return json.dumps(config_dict, indent=2, sort_keys=True, default=default) + "\n"
    
    
    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))