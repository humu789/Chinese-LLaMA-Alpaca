import copy
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import transformers
import datasets
from datasets import load_dataset, concatenate_datasets, Dataset


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=IGNORE_INDEX)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(IGNORE_INDEX),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def get_eval_dataloader_with_all_columns(trainer, eval_dataset: Optional[Dataset] = None) -> DataLoader:
    """
    Returns the evaluation [`~torch.utils.data.DataLoader`].

    Subclass and override this method if you want to inject some custom behavior.

    Args:
        eval_dataset (`torch.utils.data.Dataset`, *optional*):
            If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
            by the `model.forward()` method are automatically removed. It must implement `__len__`.
    """
    if eval_dataset is None and trainer.eval_dataset is None:
        raise ValueError("Trainer: evaluation requires an eval_dataset.")
    eval_dataset = eval_dataset if eval_dataset is not None else trainer.eval_dataset
    data_collator = trainer.data_collator

    if isinstance(eval_dataset, torch.utils.data.IterableDataset):
        if trainer.args.world_size > 1:
            eval_dataset = IterableDatasetShard(
                eval_dataset,
                batch_size=trainer.args.per_device_eval_batch_size,
                drop_last=trainer.args.dataloader_drop_last,
                num_processes=trainer.args.world_size,
                process_index=trainer.args.process_index,
            )
        return DataLoader(
            eval_dataset,
            batch_size=trainer.args.eval_batch_size,
            collate_fn=data_collator,
            num_workers=trainer.args.dataloader_num_workers,
            pin_memory=trainer.args.dataloader_pin_memory,
        )
    eval_sampler = trainer._get_eval_sampler(eval_dataset)

    return DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=trainer.args.eval_batch_size,
        collate_fn=data_collator,
        drop_last=trainer.args.dataloader_drop_last,
        num_workers=trainer.args.dataloader_num_workers,
        pin_memory=trainer.args.dataloader_pin_memory,
    )
