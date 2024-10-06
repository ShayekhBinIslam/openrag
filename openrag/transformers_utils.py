import os
import collections
import re
import shutil
import tempfile
import gc
from copy import deepcopy

import torch

from transformers.utils import logging
from transformers.pytorch_utils import id_tensor_storage
from transformers.modeling_utils import (
    set_initialized_submodules,
    expand_device_map,
    _load_state_dict_into_model,
    get_disk_only_shard_files,
    load_state_dict,
    _load_state_dict_into_meta_model,
)
from accelerate.utils import (
    find_tied_parameters,
    set_module_tensor_to_device,
    load_offloaded_weights,
    save_offload_index,
)

logger = logging.get_logger(__name__)


## Don't convert MoE Adapter to Low Bit Linear
def get_keys_to_not_convert(model):
    r"""
    An utility function to get the key of the module to keep in full precision if any For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons. For other architectures, we want
    to keep the tied weights of the model. The function will return a list of the keys of the modules to not convert in
    int8.

    Parameters:
    model (`torch.nn.Module`):
        Input model
    """
    # Create a copy of the model and tie the weights, then
    # check if it contains tied weights
    tied_model = deepcopy(
        model
    )  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0

    # Check if it is a base model
    is_base_model = not hasattr(model, model.base_model_prefix)

    # Ignore this for base models (BertModel, GPT2Model, etc.)
    if (not has_tied_params) and is_base_model:
        return []

    adapter_module = []
    for n, p in model.named_parameters():
        if "adapter" in n:
            adapter_module.append(n)

    # otherwise they have an attached head
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]

    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection) + adapter_module

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    # print(filtered_module_names)
    return filtered_module_names


@classmethod
def _load_pretrained_model(
    cls,
    model,
    state_dict,
    loaded_keys,
    resolved_archive_file,
    pretrained_model_name_or_path,
    ignore_mismatched_sizes=False,
    sharded_metadata=None,
    _fast_init=True,
    low_cpu_mem_usage=False,
    device_map=None,
    offload_folder=None,
    offload_state_dict=None,
    dtype=None,
    is_quantized=False,
    keep_in_fp32_modules=None,
):
    is_safetensors = False
    if is_quantized:
        from transformers.utils.bitsandbytes import (
            set_module_quantized_tensor_to_device,
        )

    if device_map is not None and "disk" in device_map.values():
        archive_file = (
            resolved_archive_file[0]
            if isinstance(resolved_archive_file, (list, tuple))
            else resolved_archive_file
        )
        is_safetensors = archive_file.endswith(".safetensors")
        if offload_folder is None and not is_safetensors:
            raise ValueError(
                "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
                " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
                " offers the weights in this format."
            )
        if offload_folder is not None:
            os.makedirs(offload_folder, exist_ok=True)
        if offload_state_dict is None:
            offload_state_dict = True

    is_sharded_safetensors = is_safetensors and sharded_metadata is not None
    # Retrieve missing & unexpected_keys
    model_state_dict = model.state_dict()
    expected_keys = list(model_state_dict.keys())
    prefix = model.base_model_prefix

    def _fix_key(key):
        if "beta" in key:
            return key.replace("beta", "bias")
        if "gamma" in key:
            return key.replace("gamma", "weight")
        return key

    original_loaded_keys = loaded_keys
    loaded_keys = [_fix_key(key) for key in loaded_keys]

    if len(prefix) > 0:
        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
    else:
        has_prefix_module = False
        expects_prefix_module = False

    # key re-naming operations are never done on the keys
    # that are loaded, but always on the keys of the newly initialized model
    remove_prefix_from_model = not has_prefix_module and expects_prefix_module
    add_prefix_to_model = has_prefix_module and not expects_prefix_module

    if remove_prefix_from_model:
        _prefix = f"{prefix}."
        expected_keys_not_prefixed = [
            s for s in expected_keys if not s.startswith(_prefix)
        ]
        expected_keys = [
            s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys
        ]
    elif add_prefix_to_model:
        expected_keys = [".".join([prefix, s]) for s in expected_keys]

    missing_keys = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = set(loaded_keys) - set(expected_keys)
    # Remove nonpersistent buffers from unexpected keys: they are not in the state dict but will be in the model
    # buffers
    model_buffers = {n for n, _ in model.named_buffers()}
    if remove_prefix_from_model:
        model_buffers = {
            key[len(_prefix) :] if key.startswith(_prefix) else key
            for key in model_buffers
        }
    elif add_prefix_to_model:
        model_buffers = {".".join([prefix, key]) for key in model_buffers}
    unexpected_keys = list(unexpected_keys - model_buffers)

    model.tie_weights()
    ptrs = collections.defaultdict(list)
    for name, tensor in model.state_dict().items():
        id_tensor = (
            id_tensor_storage(tensor)
            if tensor.device != torch.device("meta")
            else id(tensor)
        )
        ptrs[id_tensor].append(name)

    # These are all the pointers of shared tensors.
    tied_params = [names for _, names in ptrs.items() if len(names) > 1]

    for group in tied_params:
        if remove_prefix_from_model:
            group = [
                key[len(_prefix) :] if key.startswith(_prefix) else key for key in group
            ]
        elif add_prefix_to_model:
            group = [".".join([prefix, key]) for key in group]
        missing_in_group = [k for k in missing_keys if k in group]
        if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
            missing_keys = [k for k in missing_keys if k not in missing_in_group]

    # Some models may have keys that are not in the state by design, removing them before needlessly warning
    # the user.
    if cls._keys_to_ignore_on_load_missing is not None:
        for pat in cls._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

    if cls._keys_to_ignore_on_load_unexpected is not None:
        for pat in cls._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    # retrieve weights on meta device and put them back on CPU.
    # This is not ideal in terms of memory, but if we don't do that not, we can't initialize them in the next step
    if low_cpu_mem_usage:
        for key in missing_keys:
            if key in list(model_state_dict.keys()):
                key = key
            elif f"{prefix}.{key}" in list(model_state_dict.keys()):
                key = f"{prefix}.{key}"
            elif key.startswith(prefix) and ".".join(key.split(".")[1:]) in list(
                model_state_dict.keys()
            ):
                key = ".".join(key.split(".")[1:])
            param = model_state_dict[key]

            # upcast in fp32 if any
            target_dtype = dtype
            if (
                keep_in_fp32_modules is not None
                and dtype == torch.float16
                and any(
                    module_to_keep_in_fp32 in key
                    for module_to_keep_in_fp32 in keep_in_fp32_modules
                )
            ):
                target_dtype = torch.float32

            if param.device == torch.device("meta"):
                if not (is_quantized):
                    set_module_tensor_to_device(
                        model,
                        key,
                        "cpu",
                        torch.empty(*param.size(), dtype=target_dtype),
                    )
                else:
                    set_module_quantized_tensor_to_device(
                        model,
                        key,
                        "cpu",
                        torch.empty(*param.size(), dtype=target_dtype),
                    )

    # retrieve unintialized modules and initialize before maybe overriding that with the pretrained weights.
    if _fast_init:
        if remove_prefix_from_model:
            _loaded_keys = [f"{prefix}.{k}" for k in loaded_keys]
        elif add_prefix_to_model:
            _loaded_keys = [k[len(prefix) + 1 :] for k in loaded_keys]
        else:
            _loaded_keys = loaded_keys
        set_initialized_submodules(model, _loaded_keys)
        # This will only initialize submodules that are not marked as initialized by the line above.
        model.apply(model._initialize_weights)

    # Set some modules to fp32 if any
    if keep_in_fp32_modules is not None:
        for name, param in model.named_parameters():
            if any(
                module_to_keep_in_fp32 in name
                for module_to_keep_in_fp32 in keep_in_fp32_modules
            ):
                param = param.to(torch.float32)

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = ""
    model_to_load = model
    if (
        len(cls.base_model_prefix) > 0
        and not hasattr(model, cls.base_model_prefix)
        and has_prefix_module
    ):
        start_prefix = cls.base_model_prefix + "."
    if (
        len(cls.base_model_prefix) > 0
        and hasattr(model, cls.base_model_prefix)
        and not has_prefix_module
    ):
        model_to_load = getattr(model, cls.base_model_prefix)
        base_model_expected_keys = list(model_to_load.state_dict().keys())
        if any(
            key in expected_keys_not_prefixed and key not in base_model_expected_keys
            for key in loaded_keys
        ):
            raise ValueError(
                "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                "properly saved?"
            )
        if device_map is not None:
            device_map = {
                k.replace(f"{cls.base_model_prefix}.", ""): v
                for k, v in device_map.items()
            }

    def _find_mismatched_keys(
        state_dict,
        model_state_dict,
        loaded_keys,
        add_prefix_to_model,
        remove_prefix_from_model,
        ignore_mismatched_sizes,
    ):
        mismatched_keys = []
        if ignore_mismatched_sizes:
            for checkpoint_key in loaded_keys:
                # If the checkpoint is sharded, we may not have the key here.
                if checkpoint_key not in state_dict:
                    continue
                model_key = checkpoint_key
                if remove_prefix_from_model:
                    # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                    model_key = f"{prefix}.{checkpoint_key}"
                elif add_prefix_to_model:
                    # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                    model_key = ".".join(checkpoint_key.split(".")[1:])

                if (
                    model_key in model_state_dict
                    and state_dict[checkpoint_key].shape
                    != model_state_dict[model_key].shape
                ):
                    mismatched_keys.append(
                        (
                            checkpoint_key,
                            state_dict[checkpoint_key].shape,
                            model_state_dict[model_key].shape,
                        )
                    )
                    del state_dict[checkpoint_key]

        return mismatched_keys

    if resolved_archive_file is not None:
        folder = os.path.sep.join(resolved_archive_file[0].split(os.path.sep)[:-1])
    else:
        folder = None
    if device_map is not None and is_safetensors:
        param_device_map = expand_device_map(device_map, original_loaded_keys)

        str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
        if sharded_metadata is None:
            archive_file = (
                resolved_archive_file[0]
                if isinstance(resolved_archive_file, (list, tuple))
                else resolved_archive_file
            )
            weight_map = {p: archive_file for p in original_loaded_keys}
        else:
            weight_map = {
                p: os.path.join(folder, f)
                for p, f in sharded_metadata["weight_map"].items()
            }
        offload_index = {
            p: {"safetensors_file": f, "weight_name": p, "dtype": str_dtype}
            for p, f in weight_map.items()
            if param_device_map[p] == "disk"
        }

    if state_dict is not None:
        # Whole checkpoint
        mismatched_keys = _find_mismatched_keys(
            state_dict,
            model_state_dict,
            original_loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        )
        error_msgs = _load_state_dict_into_model(
            model_to_load, state_dict, start_prefix
        )
        offload_index = None
    else:
        # Sharded checkpoint or whole but low_cpu_mem_usage==True

        # This should always be a list but, just to be sure.
        if not isinstance(resolved_archive_file, list):
            resolved_archive_file = [resolved_archive_file]

        error_msgs = []
        mismatched_keys = []
        if not is_safetensors:
            offload_index = (
                {} if device_map is not None and "disk" in device_map.values() else None
            )
        if offload_state_dict:
            state_dict_folder = tempfile.mkdtemp()
            state_dict_index = {}
        else:
            state_dict_folder = None
            state_dict_index = None

        if is_sharded_safetensors:
            disk_only_shard_files = get_disk_only_shard_files(
                device_map, sharded_metadata=sharded_metadata
            )
            disk_only_shard_files = [
                os.path.join(folder, f) for f in disk_only_shard_files
            ]
        else:
            disk_only_shard_files = []

        if len(resolved_archive_file) > 1:
            resolved_archive_file = logging.tqdm(
                resolved_archive_file, desc="Loading checkpoint shards"
            )
        for shard_file in resolved_archive_file:
            # Skip the load for shards that only contain disk-offloaded weights when using safetensors for the offload.
            if shard_file in disk_only_shard_files:
                continue
            state_dict = load_state_dict(shard_file)

            # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
            # matching the weights in the model.
            mismatched_keys += _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )

            if low_cpu_mem_usage:
                new_error_msgs, offload_index, state_dict_index = (
                    _load_state_dict_into_meta_model(
                        model_to_load,
                        state_dict,
                        loaded_keys,
                        start_prefix,
                        expected_keys,
                        device_map=device_map,
                        offload_folder=offload_folder,
                        offload_index=offload_index,
                        state_dict_folder=state_dict_folder,
                        state_dict_index=state_dict_index,
                        dtype=dtype,
                        is_quantized=is_quantized,
                        is_safetensors=is_safetensors,
                        keep_in_fp32_modules=keep_in_fp32_modules,
                    )
                )
                error_msgs += new_error_msgs
            else:
                error_msgs += _load_state_dict_into_model(
                    model_to_load, state_dict, start_prefix
                )

            # force memory release
            del state_dict
            gc.collect()

        if offload_index is not None and len(offload_index) > 0:
            if model != model_to_load:
                # We need to add the prefix of the base model
                prefix = cls.base_model_prefix
                if not is_safetensors:
                    for weight_name in offload_index:
                        shutil.move(
                            os.path.join(offload_folder, f"{weight_name}.dat"),
                            os.path.join(offload_folder, f"{prefix}.{weight_name}.dat"),
                        )
                offload_index = {
                    f"{prefix}.{key}": value for key, value in offload_index.items()
                }
            if not is_safetensors:
                save_offload_index(offload_index, offload_folder)
                offload_index = None

        if offload_state_dict:
            # Load back temporarily offloaded state dict
            load_offloaded_weights(model_to_load, state_dict_index, state_dict_folder)
            shutil.rmtree(state_dict_folder)

    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        if "size mismatch" in error_msg:
            error_msg += "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
        raise RuntimeError(
            f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}"
        )

    if is_quantized:
        unexpected_keys = [elem for elem in unexpected_keys if "SCB" not in elem]
        missing_keys = [elem for elem in missing_keys if "SCB" not in elem]

    missing_keys = list(filter(lambda x: "adapter" not in x, missing_keys))

    if len(unexpected_keys) > 0:
        archs = [] if model.config.architectures is None else model.config.architectures
        warner = logger.warn if model.__class__.__name__ in archs else logger.info
        warner(
            f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
            f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
            " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
            " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
            f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
            " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
        )
    else:
        logger.info(
            f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
        )
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
            " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
    elif len(mismatched_keys) == 0:
        logger.info(
            f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
            f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
            " training."
        )
    if len(mismatched_keys) > 0:
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        logger.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
            f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
            " to use it for predictions and inference."
        )

    return (
        model,
        missing_keys,
        unexpected_keys,
        mismatched_keys,
        offload_index,
        error_msgs,
    )
