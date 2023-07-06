from transformers import LlamaForCausalLM
import evaluate
import datasets

perplexity = evaluate.load("perplexity", module_type="metric")
input_texts = datasets.load_dataset("wikitext",
                                    "wikitext-2-raw-v1",
                                    split="test")["text"][:50]
input_texts = [s for s in input_texts if s!='']
results = perplexity.compute(model_id='gpt2',
                             predictions=input_texts)
print(results)