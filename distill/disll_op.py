# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from .ops import QuantLinear
from .fake_quants import ZeroQuantWFakeQuantize, ZeroQuantAFakeQuantize
from torch.ao.quantization import QConfig


class DistillOp(nn.Module):

    def __init__(self, stu: nn.Module, tea: nn.Module) -> None:
        super().__init__()

        self.stu = stu
        self.tea = tea

    def forward(self, x: torch.Tensor):
        if self.training:
            assert x.shape[
                0] % 2 == 0, "batch size must be even for self distillation."
            x1, x2 = torch.split(x, x.shape[0] // 2, dim=0)

            y1 = self.stu(x2)
            with torch.no_grad():
                y2 = self.tea(x2).detach()
            y = torch.cat([y1, y2], dim=0)

            loss = self.compute_distill_loss(y1, y2)
            self.loss = loss

            return y
        else:
            return self.stu(x)

    def compute_distill_loss(self, y1, y2):
        return (y1 - y2).norm(2)


def linear2linear(module: nn.Linear):
    new_module = nn.Linear(module.in_features, module.out_features, module.bias
                           is not None)
    new_module.load_state_dict(module.state_dict())
    return new_module

def linear2quantlinear(module: nn.Linear):
    bit = 8
    quant_min = 0
    quant_max = 2 ** bit - 1
    w_fakequant=ZeroQuantWFakeQuantize.with_args(
        quant_min=quant_min, 
        quant_max=quant_max,
    )
    a_fakequant=ZeroQuantAFakeQuantize.with_args(
        quant_min=quant_min, 
        quant_max=quant_max,
    )
    qconfig = QConfig(weight=w_fakequant,
                      activation=a_fakequant)
    module.qconfig = qconfig
    new_module = QuantLinear.from_float(module)
    return new_module

def layernorm2layernorm(module: nn.LayerNorm):
    new_module = nn.LayerNorm(module.normalized_shape)
    new_module.load_state_dict(module.state_dict())
    return new_module


default_op_generator = {
    nn.Linear: linear2linear,
    nn.LayerNorm: layernorm2layernorm
}

fakequant_op_generator = {
    nn.Linear: linear2quantlinear,
}


class SelfDistillMutator():

    def __init__(self, stu_op_generator=default_op_generator) -> None:
        self.model: nn.Module = None
        self.stu_op_generator = stu_op_generator

    def prepare_from_supernet(self, model: nn.Module) -> None:
        self.model = model

        def replace_op(model: nn.Module, name: str, module: nn.Module):
            names = name.split('.')
            for sub_name in names[:-1]:
                model = getattr(model, sub_name)

            setattr(model, names[-1], module)

        for name, module in model.named_modules():
            if type(module) in self.stu_op_generator:
                stu_module = self.stu_op_generator[type(module)](module)
                distill_op = DistillOp(stu_module, module)
                replace_op(model, name, distill_op)

    def gather_distill_loss(self):
        loss = 0
        for op in self.distill_ops:
            if op.loss is not None:
                loss = loss + op.loss
        return loss

    @property
    def distill_ops(self):
        assert self.model is not None
        for module in self.model.modules():
            if isinstance(module, DistillOp):
                yield module

    @property
    def named_distill_ops(self):
        for name, module in self.model.named_modules():
            if isinstance(module, DistillOp):
                yield name, module
