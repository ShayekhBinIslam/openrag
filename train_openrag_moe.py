#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
from os.path import exists, join, isdir
import gc
import json
import math
import random
import copy
from copy import deepcopy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Callable, List, Tuple, Union, Any

import torch
from torch import nn
from torch.utils.data import Dataset
import bitsandbytes as bnb

import transformers
from transformers import Trainer, BitsAndBytesConfig, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

from datasets import load_dataset

from openrag.configuration_openrag import OpenRAGConfig
from openrag.modeling_openrag import LlamaForCausalLM

import warnings

warnings.filterwarnings("ignore")

from openrag.transformers_utils import (
    get_keys_to_not_convert,
    _load_pretrained_model,
)
import transformers.integrations
import transformers.modeling_utils
from ipdb import set_trace as stt

transformers.integrations.get_keys_to_not_convert = get_keys_to_not_convert
transformers.modeling_utils.PreTrainedModel._load_pretrained_model = (
    _load_pretrained_model
)
import GPUtil

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
context_markups = []


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    lora_r: int = field(
        default=64,
        metadata={"help": "Rank of the LoRA update matrices"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Scaling factor for LoRA"}
    )
    adapter_dim: int = field(
        default=64,
        metadata={"help": "Dimension of the adapter layers"}
    )
    moe_scaling: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for Mixture of Experts"}
    )
    num_experts: int = field(
        default=8,
        metadata={"help": "Number of experts in the Mixture of Experts layer"}
    )
    topk: int = field(
        default=2,
        metadata={"help": "Top-k value for routing or selection"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    
    data_subset: str = field(
        default="default", metadata={"help": "Training data subset."}
    )
    
    


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: str = field(default="none")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(
        default="paged_adamw_32bit"
    )  # "paged_lion_8bit", "paged_adamw_8bit", "paged_lion_32bit", "paged_adamw_32bit"
    lr_scheduler_type: str = field(
        default="constant_with_warmup"
    )  # "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )



def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [
        s + t 
        for s, t in zip(sources, targets)
    ]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        global context_markups

        context_start = False
        for j, orig_token in enumerate(label[source_len:]):
            if context_start is False and orig_token == context_markups[0]:
                context_start = True
                assert label[source_len + j] == context_markups[0]
                start_idx = j + source_len
                end_idx = None
                for k, orig_token_2 in enumerate(label[start_idx:]):
                    if orig_token_2 == context_markups[1]:
                        end_idx = start_idx + k
                if end_idx is None:
                    end_idx = start_idx + k
                else:
                    assert label[end_idx] == context_markups[1]
                label[start_idx + 1 : end_idx] = IGNORE_INDEX
                context_start = False

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, data_subset: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Loading data: {}".format(data_path))        
        dataset = load_dataset(data_path, data_subset)
        data_list = dataset['train'].to_list()

        # Preprocess Data
        logging.warning("Processing data")
        self.tokenizer = tokenizer
        self.sources = []
        self.targets = []

        for idx in range(len(data_list)):
            data = data_list[idx]
            corpus = data["corpus"]
            if corpus != "":
                # pretrain mode
                source = f"{tokenizer.bos_token}"
                self.sources.append(source)

                target = f"{corpus}{tokenizer.eos_token}"
                self.targets.append(target)
            else:
                # instruction mode
                instruction = data["instruction"]
                conversation = data["conversation"]
                if len(conversation) == 1:
                    source = ""
                    source += (
                        f"### Instruction:\n{conversation[0]['input']}\n\n### Response:\n"
                    )
                    self.sources.append(source)
                    target = f"{conversation[0]['output']}{tokenizer.eos_token}"
                    self.targets.append(target)
                

        del data_list
        gc.collect()

        logging.warning("there are {} data in dataset".format(len(self.sources)))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        source = [self.sources[i]]
        target = [self.targets[i]]
        data_dict = preprocess(source, target, self.tokenizer)
        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]

        return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        # print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        model = kwargs["model"]
        model.save_pretrained(peft_model_path)

        moe_state = {}
        for param_tensor in model.state_dict():
            if "adapter" in param_tensor:
                moe_state.update({param_tensor: model.state_dict()[param_tensor]})
    
        moe_model_path = os.path.join(checkpoint_folder, "moe_model.bin")
        torch.save(moe_state, moe_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        data_path=data_args.data_path, data_subset=data_args.data_subset,
        tokenizer=tokenizer, 
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def find_all_linear_names(model, bits=4):
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class PrinterCallback(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        GPUtil.showUtilization(all=True, attrList=None, useOldCode=True)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.ddp_find_unused_parameters = False
    set_seed(42)

    model_config = OpenRAGConfig.from_pretrained(model_args.model_name_or_path)
    model_config.pretraining_tp = 1  ## without tensor parallelism rank

    # OpenRAG Config
    model_config.moe_dtype = "bfloat16"
    model_config.lora_r = model_args.lora_r
    model_config.lora_alpha = model_args.lora_alpha
    model_config.adapter_dim = model_args.adapter_dim
    model_config.topk = model_args.topk
    model_config.moe_scaling = model_args.moe_scaling
    model_config.num_experts = model_args.num_experts
    model_config.output_router_logits = False

    # # Seq Length Extension
    # model_config.rope_scaling = {
    #     "type": "dynamic",
    #     "factor": 2,
    # }

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        cache_dir=training_args.cache_dir,
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        output_loading_info=False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    special_token_dict = {
        "additional_special_tokens": [
            "[No Retrieval]",
            "[Retrieval]",
            "[Continue to Use Evidence]",
            "[Irrelevant]",
            "[Relevant]",
            "<paragraph>",
            "</paragraph>",
            "[Utility:1]",
            "[Utility:2]",
            "[Utility:3]",
            "[Utility:4]",
            "[Utility:5]",
            "[Fully supported]",
            "[Partially supported]",
            "[No support / Contradictory]",
        ]
    }

    special_token_dict["bos_token"] = "<s>"
    special_token_dict["eos_token"] = "</s>"
    special_token_dict["unk_token"] = "<unk>"
    special_token_dict["pad_token"] = "<pad>"
    
    num_added_tokens = tokenizer.add_special_tokens(special_token_dict)
    
    global context_markups
    context_markups = []
    for token in ["<paragraph>", "</paragraph>"]:
        context_markups.append(tokenizer.convert_tokens_to_ids(token))
    
    assert (
        num_added_tokens > 10
    ), "special tokens must be added to the original tokenizers."


    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    
    lora_modules = []
    lora_modules += [
        "embed_tokens", 
        "lm_head",
    ]
    
    lora_modules += [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ]
    
    
    config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=lora_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # Zero Init
    for n, p in model.named_parameters():
        if "adapter_up" in n:
            nn.init.zeros_(p)
        if "adapter_down" in n:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        if "router" in n:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
        if "adapter" in name:
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            else:
                module = module.to(torch.float32)

    for n, p in model.named_parameters():
        if "adapter" in n:
            p.requires_grad = True

    model.config.use_cache = False
    print_trainable_parameters(model)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.add_callback(SavePeftModelCallback)
    trainer.add_callback(PrinterCallback)

    trainer.train()

    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
