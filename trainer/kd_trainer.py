from typing import Callable, Dict, List, Optional, Tuple, Union
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

class KDTraniner(Trainer):

    def __init__(
        self,
        tea_model,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(model, 
                         args, 
                         data_collator, 
                         train_dataset, 
                         eval_dataset, 
                         tokenizer, 
                         model_init, 
                         compute_metrics, 
                         callbacks, 
                         optimizers, 
                         preprocess_logits_for_metrics)
        self.tea_model = tea_model
        self._move_model_to_device(self.tea_model, self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss_label = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss_label = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss_label = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        with torch.no_grad():
            outputs_tea = self.tea_model(**inputs)
        soft_tea_probs = F.softmax(outputs_tea.logits / self.args.temperature, dim=-1)
        soft_stu_probs = F.log_softmax(outputs.logits / self.args.temperature, dim=-1)
        loss_logits = F.kl_div(input=soft_stu_probs,
                                target=soft_tea_probs,
                                reduction=self.args.reduction) * (self.args.temperature**2)

        loss = self.args.alpha_label * loss_label + self.args.alpha_logits * loss_logits

        return (loss, outputs) if return_outputs else loss
