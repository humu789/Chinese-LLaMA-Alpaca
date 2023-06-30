import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default=None, type=str, required=True)
parser.add_argument('--num_sequence', default=5, type=int)
parser.add_argument('--max_length', default=1024, type=int)
parser.add_argument('--save_file', default='./generate_data.txt', type=str)
parser.add_argument('--add_mode', action='store_true', help='add data to the save file, default to rewrite')
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')
args = parser.parse_args()

if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path, 
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        )

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size!=tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(tokenzier_vocab_size)

    if device==torch.device('cpu'):
        model.float()

    model.eval()

    vocabulary = tokenizer.get_vocab()

    with torch.no_grad():
        print("Start generating data...")
        
        save_dir, _ = os.path.split(args.save_file)
        os.makedirs(save_dir, exist_ok=True)
        
        mode = 'w'
        if args.add_mode:
            mode = 'a'
        with open(args.save_file, mode, encoding='utf-8') as file:
        
            for i in tqdm(range(args.num_sequence)):
                random_text = random.choice(list(vocabulary.keys()))
                inputs = tokenizer(random_text, return_tensors="pt", add_special_tokens=False)
                top_1_samples = np.random.randint(3, 6)
                
                top_1_outputs = model.generate(
                    input_ids=inputs["input_ids"][:,-1:].to(device),
                    max_length=1+top_1_samples,
                    do_sample=False,
                    eos_token_id=None)
                
                generate_outputs = model.generate(
                    input_ids=top_1_outputs,
                    max_length=args.max_length,
                    do_sample=True,
                    top_k=0,
                    eos_token_id=model.config.eos_token_id)
                
                top_1_text = tokenizer.decode(top_1_outputs[0], skip_special_tokens=True)
                output_text = tokenizer.decode(generate_outputs[0], skip_special_tokens=True)
                
                if i < 5:
                    print(f'Top_1_Tokens: {top_1_samples}')
                    print(f'Top_1_Text: {top_1_text}')
                    print(f'All_Tokens: {len(generate_outputs[0])}')
                    print(f'All_Text: {output_text}')
                    print('=='*30)

                
                file.write(output_text + '\n')
        
        print("Finish generating data!")
