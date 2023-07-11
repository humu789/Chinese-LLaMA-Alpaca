from transformers import TrainerCallback
from datasets import load_dataset, load_metric
from tqdm import tqdm
import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import evaluate
from evaluate import logging

class PerplexityEvalCallback(TrainerCallback):
    def __init__(self, trainer, data_loader):
        self.trainer = trainer
        self.data_loader = data_loader

    def compute_perplexity(self, model):
        model.eval()
        ppls = 0
        loss_fct = CrossEntropyLoss(reduction="none")
        use_cache = model.config.use_cache
        model.config.use_cache = False
        
        for batch in tqdm(self.data_loader):
            (_, logits, labels) = self.trainer.prediction_step(model, batch, prediction_loss_only=False)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))        
            ppl = torch.exp(loss)
            ppls += ppl
        
        model.config.use_cache = use_cache
        return ppls
        # return {"mean_perplexity": np.mean(ppls)}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        ppls = self.compute_perplexity(model)
        dist.barrier()
        dist.all_reduce(ppls, op=dist.ReduceOp.AVG)
        dist.barrier()
        perplexity_score = ppls.item() / len(self.data_loader)
        if int(os.environ["RANK"]) == 0:
            print('#'*100)
            print(f"Perplexity at beginning: {perplexity_score}")
            print('#'*100)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        ppls = self.compute_perplexity(model)
        dist.barrier()
        dist.all_reduce(ppls, op=dist.ReduceOp.AVG)
        dist.barrier()
        perplexity_score = ppls.item() / len(self.data_loader)
        if int(os.environ["RANK"]) == 0:
            print('#'*100)
            print(f"Perplexity at end: {perplexity_score}")
            print('#'*100)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        ppls = self.compute_perplexity(model)
        dist.barrier()
        dist.all_reduce(ppls, op=dist.ReduceOp.AVG)
        dist.barrier()
        perplexity_score = ppls.item() / len(self.data_loader)
        if int(os.environ["RANK"]) == 0:
            print('#'*100)
            print(f"Perplexity after evaluation: {perplexity_score}")
            print('#'*100)

