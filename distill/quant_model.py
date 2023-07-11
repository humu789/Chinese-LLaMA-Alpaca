from transformers import PreTrainedModel
from torch import nn
# from torch.ao.quantization import QConfig
from typing import Optional
from .ops import QuantLinear as QLinear
from torch.ao.quantization import (QConfig, enable_fake_quant, enable_observer,
                                   disable_fake_quant, disable_observer)
from torch.ao.nn.qat import Linear as WQLiear


class HfLlamaWrapper(nn.Module):

    def __init__(self, 
                 reference,
                 qconfig: QConfig, 
                 kv_qconfig: QConfig,
                 weight_only=True,
                 kv_module_names=[],
                 skip_module_names=[],
                 spec_bit2_layers=[],
                 spec_bit2_qconfig=None):
        super().__init__()
        self.reference = reference
        self.qconfig = qconfig
        self.kv_qconfig = kv_qconfig
        self.kv_module_names = kv_module_names
        self.skip_module_names = skip_module_names
        self.spec_bit2_layers = spec_bit2_layers
        self.spec_bit2_qconfig = spec_bit2_qconfig
        if len(spec_bit2_layers) > 0:
            assert spec_bit2_qconfig is not None
        
        if weight_only:
            self._inplace_weight_only_qlinear(self.reference, 
                                            self.qconfig,
                                            self.kv_qconfig,
                                            self.kv_module_names,
                                            self.skip_module_names)
        else:
            self._inplace_qlinear(self.reference, 
                                self.qconfig,
                                self.kv_qconfig,
                                self.kv_module_names,
                                self.skip_module_names)

    def _inplace_weight_only_qlinear(self,
                                     module,
                                     qconfig,
                                     kv_qconfig,
                                     kv_module_names,
                                     skip_module_names):
        def travase(m, qconfig, kv_qconfig, kv_module_names, skip_module_names=[], prefix=''):

            for name, child in m.named_children():

                full_child_name = f'{prefix}.{name}' if len(prefix) else name
                if full_child_name.startswith('model.layers'):
                    layer_index = name.split('.')[2]
                    if layer_index in self.spec_bit2_layers:
                        is_bit2 = True
                    else:
                        is_bit2 = False
                if isinstance(child,
                              nn.Linear) and name not in skip_module_names:
                    if name in kv_module_names:
                        child.qconfig = kv_qconfig
                        qlinear = QLinear.from_float(child)
                        print(f'Convert {full_child_name} to QLiear')
                    else:
                        if is_bit2:
                            child.qconfig = self.spec_bit2_qconfig
                            print(f'Convert {full_child_name} to WQLiear for 2bit')
                        else:
                            child.qconfig = qconfig
                            print(f'Convert {full_child_name} to WQLiear')
                        qlinear = WQLiear.from_float(child)
                    
                    setattr(m, name, qlinear) 
                else:
                    travase(child, qconfig, kv_qconfig, kv_module_names, skip_module_names, full_child_name)

        travase(module, qconfig, kv_qconfig, kv_module_names, skip_module_names)

    def _inplace_qlinear(self, 
                         module, 
                         qconfig,
                         kv_qconfig,
                         kv_module_names,
                         skip_module_names):

        def travase(m, qconfig, kv_qconfig, kv_module_names, skip_module_names=[], prefix=''):

            for name, child in m.named_children():

                full_child_name = f'{prefix}.{name}' if len(prefix) else name
                if isinstance(child,
                              nn.Linear) and name not in skip_module_names:
                    if name in kv_module_names:
                        child.qconfig = kv_qconfig
                    else:
                        child.qconfig = qconfig
                    qlinear = QLinear.from_float(child)
                    setattr(m, name, qlinear)
                    print(f'Convert {full_child_name} to QLinear')
                else:
                    travase(child, qconfig, kv_qconfig, kv_module_names, skip_module_names, full_child_name)

        travase(module, qconfig, kv_qconfig, kv_module_names, skip_module_names)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.reference(input_ids, attention_mask, position_ids,
                              past_key_values, inputs_embeds, labels,
                              use_cache, output_attentions,
                              output_hidden_states, return_dict)

    def train(self, training):
        super().train(training)

        if training:
            enable_fake_quant(self)
            enable_observer(self)
        else:
            enable_fake_quant(self)
            disable_observer(self)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.reference, name)