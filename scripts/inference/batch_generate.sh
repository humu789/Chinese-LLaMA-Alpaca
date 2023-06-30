#!/bin/sh

model_name_or_path="/nvme/share_data/llama_ckpts/huggingface/7B"
num_sequence=10
max_length=1024
save_dir="/home/humu/data/llm-qat-mini/"
save_file_name="llama7b_generate_data_mini"

gpus=("0")

for gpu in ${gpus[@]}; do
    save_file="${save_dir}${save_file_name}_${gpu}.txt"
    nohup python generate_data.py \
    --model_name_or_path ${model_name_or_path} \
    --num_sequence ${num_sequence} \
    --max_length ${max_length} \
    --save_file ${save_file} \
    --add_mode \
    --gpus ${gpu} >> mini.out &

    sleep 1
done

wait