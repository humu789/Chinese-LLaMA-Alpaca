lr=2e-5

pretrained_model="/nvme/share_data/llama_ckpts/huggingface/7B" #"decapoda-research/llama-7b-hf"
dataset_dir="/home/humu/data/llm-qat-try"
data_cache="/home/humu/data/llm-qat-try_cache"
per_device_train_batch_size=1
per_device_eval_batch_size=1
training_steps=50000
gradient_accumulation_steps=1
output_dir="/mnt/175/humu/llm-qat/llm-qat-log_skip-lmhead_freeze-layer2-31_w4-g128-a16-kv4_lr2e-5"
# gptq_ckpt="/home/humu/ckpts/llama7b-4bit-128g-official.pt"
# resume_from_checkpoint="/nvme/humu/llm-qat/llm-qat-log_skip-lmhead_w4/checkpoint-50000"
# --resume_from_checkpoint ${resume_from_checkpoint} \
# --test_gptq \
# --gptq_ckpt ${gptq_ckpt} \
# --gptq_bits 2 \
# --freeze_layers None \
# --kv_module_names None \
# --freeze_layers None \
low_cpu_mem_usage=True

deepspeed_config_file=ds_zero2_no_offload.json

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nnodes 1 --nproc_per_node 4 --master_port 10000 run_llm_qat.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train True \
    --do_eval True \
    --do_ppl_test \
    --do_mmlu_eval \
    --w_bit 4 \
    --w_scheme per-token \
    --evaluation_strategy steps \
    --eval_steps 1000 \
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
    --save_steps 2000 \
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
    --low_cpu_mem_usage ${low_cpu_mem_usage} >> ./logs/llm-qat-log_skip-lmhead_freeze-layer2-31_w4-g128-a16-kv4_lr2e-5.txt &
