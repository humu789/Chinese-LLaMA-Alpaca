from transformers import TrainerCallback
from datasets import load_dataset, load_metric
from tqdm import tqdm
import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
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
        ppls = []
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
            ppls.append(ppl.item())
        
        model.config.use_cache = use_cache

        return {"mean_perplexity": np.mean(ppls)}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        perplexity_score = self.compute_perplexity(model)
        print(f"Perplexity at beginning: {perplexity_score}")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        perplexity_score = self.compute_perplexity(model)
        print(f"Perplexity at end: {perplexity_score}")

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        perplexity_score = self.compute_perplexity(model)
        print(f"Perplexity after evaluation: {perplexity_score}")

