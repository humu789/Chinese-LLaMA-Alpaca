lr=2e-5

pretrained_model="/nvme/share_data/llama_ckpts/huggingface/7B" #"decapoda-research/llama-7b-hf"
dataset_dir="/home/humu/data/llm-qat-try"
data_cache="/home/humu/data/llm-qat-try_cache"
per_device_train_batch_size=1
per_device_eval_batch_size=1
training_steps=50000
gradient_accumulation_steps=1
output_dir="/nvme/humu/llm-qat/llm-qat-try"
low_cpu_mem_usage=True

deepspeed_config_file=ds_zero2_no_offload.json

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes 1 --nproc_per_node 4 --master_port 10004 run_clm_pt_wo_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 1 \
    --save_steps 1000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --low_cpu_mem_usage ${low_cpu_mem_usage}
