export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20119 --master_addr=localhost torchrun_main_HTMuon.py \
    --model_config configs/llama_60m.json \
    --optimizer truncated_sgd --truncate_fraction 0.05 \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.01\
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 5000 \
    --warmup_steps 500 \
    --weight_decay 0.1\
    --dtype bfloat16 \
    --eval_every 500 \
    --wandb_name 'Your_WandB_Name_Here' \
    --target_eval_tokens 10_000_000 \
    --save_dir outputs/llama_60m_truncated_sgd \
    --save_every 5000
