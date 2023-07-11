from transformers import LlamaForCausalLM
import evaluate
import datasets
from deepspeed.compression.compress import init_compression

model_path = '/nvme/share_data/llama_ckpts/huggingface/7B'
model = LlamaForCausalLM.from_pretrained(model_path)
freeze_layers = ['2', '31']
for name, param in model.named_parameters():
    if name.startswith('model.layers'):
        layer_index = name.split('.')[2]
        if layer_index in freeze_layers:
            print(name, param.shape)

# perplexity = evaluate.load("perplexity", module_type="metric")
# input_texts = datasets.load_dataset("wikitext",
#                                     "wikitext-2-raw-v1",
#                                     split="test")["text"][:50]
# input_texts = [s for s in input_texts if s!='']
# results = perplexity.compute(model_id='gpt2',
#                              predictions=input_texts)
# print(results)