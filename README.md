# Open-RAG: Enhanced Retrieval Augmented Reasoning with Open-Source Large Language Models

Official repository for the paper [Open-RAG: Enhanced Retrieval Augmented Reasoning with Open-Source Large Language Models]().

[Models (Coming soon)]() | [Paper](https://arxiv.org/abs/2410.01782) | [Training data](https://huggingface.co/datasets/shayekh/openrag_train_data) | [Evaluation Data (Coming soon)]()

## Training 

### OpenRAG-7B-8x135M

```sh
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29506 \
  train_openrag_moe.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --data_path shayekh/openrag_train_data --data_subset moe \
  --output_dir ./checkpoints/ \
  --bf16 True --tf32 True --fp16 False \
  --model_max_length 4096 \
  --num_train_epochs 2 --gradient_accumulation_steps 8 \
  --per_device_train_batch_size 4 \
  --evaluation_strategy "no" --save_strategy "epoch" \
  --logging_strategy "steps" --report_to tensorboard --logging_steps 1 \
  --learning_rate 2e-4 --adam_beta2 0.999 \
  --lr_scheduler_type constant_with_warmup \
  --max_grad_norm 0.3 --weight_decay 0.0 --warmup_steps 200 \
  --adapter_dim 512 --moe_scaling 0.25 --num_experts 8 --topk 2
```


### OpenRAG-13B-8x213M

```sh
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29506 \
  train_openrag_moe.py \
  --model_name_or_path meta-llama/Llama-2-13b-hf \
  --data_path shayekh/openrag_train_data --data_subset moe \
  --output_dir ./checkpoints/ \
  --bf16 True --tf32 True --fp16 False \
  --model_max_length 4096 \
  --num_train_epochs 2 --gradient_accumulation_steps 8 \
  --per_device_train_batch_size 4 \
  --evaluation_strategy "no" --save_strategy "epoch" \
  --logging_strategy "steps" --report_to tensorboard --logging_steps 1 \
  --learning_rate 1e-4 --adam_beta2 0.999 \
  --lr_scheduler_type constant_with_warmup \
  --max_grad_norm 0.3 --weight_decay 0.0 --warmup_steps 200 \
  --adapter_dim 512 --moe_scaling 0.25 --num_experts 8 --topk 2
```

