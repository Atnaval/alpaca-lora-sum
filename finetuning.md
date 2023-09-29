OVERFIT

```bash
OMP_NUM_THREADS=16 WORLD_SIZE=6 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 --master_port=1234 finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --data_path 'llama-dataset-2023-05-03.json' \
    --output_dir './model' \
    --num_epochs=3 \
    --batch_size 128 \
    --val_set_size 2000 \
    --cutoff_len=1024 \
    --group_by_length \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8 \
    --learning_rate=3e-4 \
    --prompt_template_name 'sum'
```

OMP_NUM_THREADS=4 WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 --master_port=1234

OMP_NUM_THREADS=16 WORLD_SIZE=6 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 --master_port=1234 finetune_generic.py \
    --data_path 'llama-dataset-2023-05-03_10k.json' \
    --output_dir './model' \
    --num_epochs=3 \
    --batch_size 64 \
    --val_set_size 0 \
    --cutoff_len=768 \
    --group_by_length \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_alpha=16 \
    --lora_r=16 \
    --micro_batch_size=1 \
    --learning_rate=3e-4 \
    --train_on_inputs=True \
    --prompt_template_name 'sum'

30B
```bash
OMP_NUM_THREADS=16 WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1234 finetune.py \
    --base_model='decapoda-research/llama-30b-hf' \
    --data_path 'llama-dataset-2023-05-03.json' \
    --output_dir './model' \
    --num_epochs=3 \
    --batch_size 64 \
    --val_set_size 0 \
    --cutoff_len=786 \
    --group_by_length \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --micro_batch_size=8 \
    --learning_rate=3e-4 \
    --prompt_template_name 'sum'
```

torchrun --nproc_per_node=2 finetune.py  --base_model='decapoda-research/llama-7b-hf'     --data_path 'llama-dataset-2023-05-03.json'     --output_dir './model'     --num_epochs=3     --batch_size 64     --val_set_size 0     --cutoff_len=1024     --group_by_length     --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]'     --lora_r=16     --lora_alpha=16     --lora_dropout=0.05     --micro_batch_size=2     --learning_rate=3e-4     --prompt_template_name 'sum'

torchrun --nproc_per_node=2 finetune_generic.py  --data_path 'llama-dataset-2023-05-03.json'     --output_dir './model'     --num_epochs=3     --batch_size 64     --val_set_size 0     --cutoff_len=1024     --group_by_length     --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]'     --lora_r=16     --lora_alpha=16     --lora_dropout=0.05     --micro_batch_size=2     --learning_rate=3e-4     --prompt_template_name 'sum'