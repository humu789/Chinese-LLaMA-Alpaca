#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import numpy as np
import math
import time
import os
import sys
import copy
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import datasets
import torch
import torch.distributed as dist
from datasets import load_dataset, concatenate_datasets

from evaluate import load
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from sklearn.metrics import accuracy_score
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import sys 
sys.path.append('../..')
from trainer import KDTrainer, PerplexityEvalCallback
from distill.quant_model import HfLlamaWrapper

from utils import (DataCollatorForCausalLM, smart_tokenizer_and_embedding_resize,
                    get_eval_dataloader_with_all_columns, DEFAULT_PAD_TOKEN, IGNORE_INDEX)


def accuracy(predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    import torch

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError: # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={
            "help": (
                "DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True`"
            )
        },
    )


    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

@dataclass
class MyTrainingArguments(TrainingArguments):
    debug_mode: Optional[bool] = field(default=False)
    do_ppl_test: bool = field(default=False)
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    test_fp16: bool = field(default=False)
    test_gptq: bool = field(default=False)
    gptq_ckpt: Optional[str] = field(default=None)
    gptq_bits: int = field(default=4)

    # KD args
    alpha_label: Optional[float] = field(default=0.)
    alpha_logits: Optional[float] = field(default=1.)
    temperature: Optional[float] = field(default=1.)
    reduction: Optional[str] = field(default='batchmean')
    
    # Quantization args
    w_bit: Optional[int] = field(default=4)
    a_bit: Optional[int] = field(default=8)
    kv_bit: Optional[int] = field(default=4)
    ## choose only from ('per-channel', 'per-tensor', 'per-group')
    w_scheme: Optional[str] = field(default='per-channel')
    ## choose only from ('per-channel', 'per-token', 'per-tensor')
    a_scheme: Optional[str] = field(default='per-token')
    kv_scheme: Optional[str] = field(default='per-token')
    ## True or False
    w_symmetric: Optional[bool] = field(default=True)
    a_symmetric: Optional[bool] = field(default=True)
    kv_symmetric: Optional[bool] = field(default=True)

    def default_kv_module_names():
        return ['k_proj', 'v_proj']
    # kv's module name in the model
    kv_module_names: Optional[List[str]] = field(default_factory=default_kv_module_names)
    def default_skip_module_names():
        return ['lm_head']
    # skip replacing linear to quantlinear in specified modules. 
    skip_module_names: Optional[List[str]] = field(default_factory=default_skip_module_names)
    def default_freeze_layers():
        return ['2', '31']
    # freeze whose layers, not to quantize them.
    freeze_layers: Optional[List[str]] = field(default_factory=default_freeze_layers)
    def default_spec_bit2_layers():
        return ['4']
    # specify whose layers to 2 bit
    spec_bit2_layers: Optional[List[str]] = field(default_factory=default_spec_bit2_layers)
    weight_only: bool = field(default=True)

logger = logging.getLogger(__name__)



def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    os.makedirs(training_args.output_dir, exist_ok=True)
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        lm_datasets = []
        path = Path(data_args.dataset_dir)
        files = [file.name for file in path.glob("*.txt")]
        if training_args.debug_mode is True:
            files = [files[0]]
        for idx, file in enumerate(files):
            data_file = os.path.join(path, file)
            filename = ''.join(file.split(".")[:-1])
            cache_path = os.path.join(data_args.data_cache_dir, filename)
            os.makedirs(cache_path, exist_ok=True)
            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'training datasets-{filename} has been loaded from disk')
            except Exception:
                cache_dir = os.path.join(data_args.data_cache_dir, filename+"_text")
                os.makedirs(cache_dir, exist_ok=True)
                raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
                logger.info(f"{file} has been loaded")
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names = {k: os.path.join(cache_dir, f'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names = {k: os.path.join(cache_dir, f'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {block_size}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
            if idx == 0:
                lm_datasets = processed_dataset['train']
            else:
                assert lm_datasets.features.type == processed_dataset["train"].features.type
                lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])

        lm_datasets = lm_datasets.train_test_split(test_size = data_args.validation_split_percentage)

    if training_args.do_train:
        train_dataset = lm_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    if training_args.do_eval:
        eval_dataset = lm_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("eval example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))

    if model_args.model_name_or_path:
        if not training_args.test_gptq:
            torch_dtype = (
                model_args.torch_dtype
                if model_args.torch_dtype in ["auto", None]
                else getattr(torch, model_args.torch_dtype)
            )
            model = LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage
            )
        else:
            sys.path.append('../../GPTQ-for-LLaMa')
            from llama_inference import load_quant
            model = load_quant(model_args.model_name_or_path, training_args.gptq_ckpt, training_args.gptq_bits, 128, eval=False, fused_mlp=False, warmup_autotune=False)
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # model_vocab_size = model.get_output_embeddings().weight.size(0)
    # if not (
    #    (model_vocab_size==32000 and len(tokenizer)==49953) or \
    #    (model_vocab_size==32000 and len(tokenizer)==32000) or \
    #    (model_vocab_size==49953 and len(tokenizer)==49953) or \
    #    (model_vocab_size==49954 and len(tokenizer)==49954)
    # ):
    #     raise ValueError(
    #         f"The combination of base model (size: {model_vocab_size}) and tokenizer (size: {len(tokenizer)}) is not a valid configuration. Please check our project wiki for further information. \n"
    #         "Valid configurations (base model / tokenizer):\n"
    #         "- Continue pre-training original LLaMA: 32000 / 32000 \n"
    #         "- Pre-training Chinese LLaMA based on original LLaMA: 32000 / 49953 \n"
    #         "- Continue pre-training Chinese LLaMA: 49953 / 49953 \n"
    #         "- Continue pre-training Chinese Alpaca: 49954 / 49954 \n")

    model.resize_token_embeddings(len(tokenizer))

    tea_model = copy.deepcopy(model)

    for name, param in model.named_parameters():
        if name.startswith('model.layers'):
            layer_index = name.split('.')[2]
            if layer_index in training_args.freeze_layers:
                param.requires_grad = False

    from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
    from torch.ao.quantization.fake_quantize import FakeQuantize
    from torch.ao.quantization import QConfig
    from distill.fake_quants import ZeroQuantAFakeQuantize, ZeroQuantWFakeQuantize
    from distill.observers import DyanamicPerChannelMinMaxObserver
    
    # w fakequant setting
    if training_args.w_scheme in ['per-channel', 'per-tensor']:
        w_observer_cls = PerChannelMinMaxObserver if training_args.w_scheme == 'per-channel' else MinMaxObserver
        if training_args.w_scheme in ['per-channel']:
            if training_args.w_symmetric:
                w_qscheme = torch.per_channel_symmetric
            else:
                w_qscheme = torch.per_channel_affine
        else:
            if training_args.w_symmetric:
                w_qscheme = torch.per_tensor_symmetric
            else:
                w_qscheme = torch.per_tensor_affine
        w_fakequant = FakeQuantize.with_args(
            observer=w_observer_cls,
            dtype=torch.qint8,
            quant_min=-2**(training_args.w_bit-1),
            quant_max=2**(training_args.w_bit-1)-1,
            qscheme=w_qscheme)
    else:
        w_fakequant = ZeroQuantWFakeQuantize.with_args(
            dtype=torch.qint8,
            quant_min=-2**(training_args.w_bit-1),
            quant_max=2**(training_args.w_bit-1)-1,
        )
    
    # a fakequant setting
    if training_args.a_scheme == 'per-token':
        a_fakequant_cls = ZeroQuantAFakeQuantize
        a_observer_cls = DyanamicPerChannelMinMaxObserver
    elif training_args.a_scheme == 'per-channel':
        a_fakequant_cls = FakeQuantize
        a_observer_cls = PerChannelMinMaxObserver
    else:
        a_fakequant_cls = FakeQuantize
        a_observer_cls = MinMaxObserver
    if training_args.a_scheme in ['per-channel', 'per-token']:
        if training_args.a_symmetric:
            a_qscheme = torch.per_channel_symmetric
        else:
            a_qscheme = torch.per_channel_affine
    else:
        if training_args.a_symmetric:
            a_qscheme = torch.per_tensor_symmetric
        else:
            a_qscheme = torch.per_tensor_affine
    a_fakequant = a_fakequant_cls.with_args(
        observer=a_observer_cls,
        dtype=torch.qint8,
        quant_min=-2**(training_args.a_bit-1),
        quant_max=2**(training_args.a_bit-1)-1,
        qscheme=a_qscheme)

    # kv fakequant setting
    if training_args.kv_scheme == 'per-token':
        kv_fakequant_cls = ZeroQuantAFakeQuantize
        kv_observer_cls = DyanamicPerChannelMinMaxObserver
    elif training_args.kv_scheme == 'per-channel':
        kv_fakequant_cls = FakeQuantize
        kv_observer_cls = PerChannelMinMaxObserver
    else:
        kv_fakequant_cls = FakeQuantize
        kv_observer_cls = MinMaxObserver
    if training_args.kv_scheme in ['per-channel', 'per-token']:
        if training_args.kv_symmetric:
            kv_qscheme = torch.per_channel_symmetric
        else:
            kv_qscheme = torch.per_channel_affine
    else:
        if training_args.kv_symmetric:
            kv_qscheme = torch.per_tensor_symmetric
        else:
            kv_qscheme = torch.per_tensor_affine
    kv_fakequant = kv_fakequant_cls.with_args(
        observer=kv_observer_cls,
        dtype=torch.qint8,
        quant_min=-2**(training_args.kv_bit-1),
        quant_max=2**(training_args.kv_bit-1)-1,
        qscheme=kv_qscheme)
    
    qconfig = QConfig(weight=w_fakequant, activation=a_fakequant)
    kv_qconfig = QConfig(weight=w_fakequant, activation=kv_fakequant)
    
    if training_args.test_gptq:
        stu_model = model
    else:
        if training_args.test_fp16:
            stu_model = model
        else:
            spec_bit2_fakequant = ZeroQuantWFakeQuantize.with_args(
                dtype=torch.qint8,
                quant_min=-2**(2-1),
                quant_max=2**(2-1)-1,
            )
            spec_bit2_qconfig = QConfig(weight=spec_bit2_fakequant, activation=a_fakequant)
            stu_model = HfLlamaWrapper(model,
                                    qconfig=qconfig,
                                    kv_qconfig=kv_qconfig,
                                    weight_only=training_args.weight_only,
                                    kv_module_names=training_args.kv_module_names,
                                    skip_module_names=training_args.skip_module_names,
                                    spec_bit2_layers=training_args.spec_bit2_layers,
                                    spec_bit2_qconfig=spec_bit2_qconfig)
        # from deepspeed.compression.compress import init_compression
        # stu_model = init_compression(model, training_args.deepspeed)

    # Initialize our Trainer
    trainer = KDTrainer(
        tea_model=tea_model,
        model=stu_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    if training_args.do_ppl_test:
        # preprocess test dataset
        raw_dataset_test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        tokenized_dataset_test = raw_dataset_test.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns="text",
            desc="Running tokenizer on dataset",
        )
        test_dataset = tokenized_dataset_test.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        test_dataloader = trainer.get_eval_dataloader(test_dataset)
        ppl_callback = PerplexityEvalCallback(trainer, test_dataloader)
        trainer.add_callback(ppl_callback)
    
    if training_args.do_mmlu_eval:
        if training_args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': '/home/humu/Chinese-LLaMA-Alpaca/data/mmlu/zero_shot_mmlu_test.json',
                'test': '/home/humu/Chinese-LLaMA-Alpaca/data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif training_args.mmlu_dataset == 'mmlu' or training_args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': '/home/humu/Chinese-LLaMA-Alpaca/data/mmlu/five_shot_mmlu_val.json',
                'test': '/home/humu/Chinese-LLaMA-Alpaca/data/mmlu/five_shot_mmlu_test.json',
            })
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[training_args.mmlu_split]
        if training_args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(training_args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        import evaluate
        from tqdm import tqdm
        accuracy = evaluate.load("accuracy", experiment_id=str(time.time_ns()))
        class MMLUEvalCallback(transformers.TrainerCallback):
            def _do_test_mmlu(self, args, state, control, model, **kwargs):
                ori_data_collator = trainer.data_collator
                trainer.data_collator = DataCollatorForCausalLM(
                    tokenizer=tokenizer,
                    source_max_len=2048,
                    target_max_len=256,
                    train_on_source=False,
                    predict_with_generate=False,
                )
                data_loader = get_eval_dataloader_with_all_columns(trainer, mmlu_dataset)
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
                    # print(f'loss: {loss.shape}')
                    # print(f'logits: {logits[0].shape}')
                    # print(f'labels: {labels}')
                    # There are two tokens, the output, and eos token.
                    logits, kv_cache = logits
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id-1, abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 1)[:,0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss
                # Extract results by subject.
                dist.barrier()
                dist.all_reduce(loss_mmlu, op=dist.ReduceOp.AVG)
                dist.barrier()
                results = {'mmlu_loss':loss_mmlu.item()/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                for s,p,r in zip(subject, preds, refs):
                    subjects[s]['preds'].append(p)
                    subjects[s]['refs'].append(r)
                subject_sum = {}
                subject_count = {}
                subject_scores = []
                for subject in subjects:
                    subject_sum = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds'],
                        normalize=False
                    )['accuracy']
                    subject_sum = torch.tensor(subject_sum, device=trainer.model.device)
                    subject_count = torch.tensor(len(subjects[subject]['refs']), device=trainer.model.device)
                    dist.barrier()
                    dist.all_reduce(subject_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(subject_count, op=dist.ReduceOp.SUM)
                    dist.barrier()
                    if subject_count > 0:
                        subject_score = (subject_sum / subject_count).item()
                        results[f'mmlu_{training_args.mmlu_split}_accuracy_{subject}'] = subject_score
                        subject_scores.append(subject_score)
                results[f'mmlu_{training_args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator = ori_data_collator
            def on_train_begin(self, args, state, control, model, **kwargs):
                # if tokenizer._pad_token is None:
                #     smart_tokenizer_and_embedding_resize(
                #         special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                #         tokenizer=tokenizer,
                #         model=model,
                #     )
                logger.info("Test MMLU before training:")
                self._do_test_mmlu(args, state, control, model, **kwargs)
            def on_train_end(self, args, state, control, model, **kwargs):
                logger.info("Test MMLU after training:")
                self._do_test_mmlu(args, state, control, model, **kwargs)
            def on_evaluate(self, args, state, control, model, **kwargs):
                self._do_test_mmlu(args, state, control, model, **kwargs)

        trainer.add_callback(MMLUEvalCallback)


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()
