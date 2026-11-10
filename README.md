# HTMuon: Improving Muon via Heavy-Tailed Spectral Correction


## Update
- [x] ( April 2026) Based on [Su's blog](https://kexue.fm/archives/11654), we have implemented the first accelerated version of HTMuon based on streaming power iteration, and more optimized versions will be released in the future. Thanks to Jianlin Su for the great work!

## Installation
### Setup

Our repository is built on top of [Galore](https://github.com/jiaweizzhao/GaLore), [MARS](https://github.com/AGI-Arena/MARS) and [AlphaDecay](https://github.com/hed-ucas/AlphaDecay). You can configure the environment using the following command lines:
```bash
conda create -p /data/project/le-lab/conda_env/htmuon python=3.9 -y
conda activate /data/project/le-lab/conda_env/htmuon
pip install -r requirements.txt
```

### Prepare Dataset

We utilized the publicly available C4 dataset, OpenWebText dataset, CIFAR-10, CIFAR-100 and ImageNet-1K dataset, all of which can be accessed and downloaded from their respective official websites.

## Usage

Additional pretraining optimizers (Spectrally Targeted Muon, Kaon, Freon SVD,
and TruncatedSGD), with Llama-60M launchers:
[configuration and implementation notes](llama_pretraining/PAPER_OPTIMIZERS.md).


#### Pretraining LLama-60M on C4 (By streaming power iteration)
```bash
export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20119 --master_addr=localhost torchrun_main_HTMuon.py \
    --model_config configs/llama_60m.json \
    --optimizer htmuon \ #or acceleration use "htmuon_ns" / "htmuon_interval" (with interval=5)
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --power 0.125 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.1\
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name 'Your_WandB_Name_Here' \
    --target_eval_tokens 10_000_000 \
    --save_every 10000
```

#### Pretraining LLama-135M on C4 (By streaming power iteration)
```bash
export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20119 --master_addr=localhost torchrun_main_HTMuon.py \
    --model_config configs/llama_135m.json 
    --optimizer htmuon_stream \  #or acceleration use "htmuon_ns" / "htmuon_interval" (with interval=5)
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.1\
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name 'Your_WandB_Name_Here' \
    --target_eval_tokens 10_000_000 \
    --save_every 10000
```


#### Pretraining LLama-60M on C4 (By SVD)
```bash
export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20119 --master_addr=localhost torchrun_main_HTMuon.py \
    --model_config configs/llama_60m.json \
    --optimizer htmuon \ #or acceleration use "htmuon_ns" / "htmuon_interval" (with interval=5)
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --power 0.125 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.1\
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name 'Your_WandB_Name_Here' \
    --target_eval_tokens 10_000_000 \
    --save_every 10000
```



#### Pretraining LLama-135M on C4 (By SVD)
```bash
export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20119 --master_addr=localhost torchrun_main_HTMuon.py \
    --model_config configs/llama_135m.json 
    --optimizer htmuon \  #or acceleration use "htmuon_ns" / "htmuon_interval" (with interval=5)
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.1\
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name 'Your_WandB_Name_Here' \
    --target_eval_tokens 10_000_000 \
    --save_every 10000
```

#### Pretraining LLama-350M on C4
```bash
export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20119 --master_addr=localhost torchrun_main_HTMuon.py \
    --model_config configs/llama_350m.json \
    --optimizer htmuon \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 5e-3\
    --power 0.125 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0.1\
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name 'Your_WandB_Name_Here' \
    --target_eval_tokens 10_000_000 \
    --save_every 10000
```

#### Pretraining LLama-1B on C4
```bash
export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20119 --master_addr=localhost torchrun_main_HTMuon.py \
    --model_config configs/llama_1b.json \
    --optimizer htmuon_interval \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 5e-3\
    --power 0.125 \
    --interval 5 \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 90000 \
    --warmup_steps 9000 \
    --weight_decay 0.1\
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name 'Your_WandB_Name_Here' \
    --target_eval_tokens 10_000_000 \
    --save_every 10000
```

### Acknowledgement
This repository is build upon the [Galore](https://github.com/jiaweizzhao/GaLore),[MARS](https://github.com/AGI-Arena/MARS)  and [AlphaDecay](https://github.com/hed-ucas/AlphaDecay) repositories. Thanks for their great work!

## Experiment Running
-  Edit those file for sbatch request:
- /data/project/le-lab/Soren/llama_pretraining/scripts/1B/HTMuonInterval.sh
- /data/project/le-lab/Soren/llama_pretraining/scripts/1B/Soren.sh


## Uploading the model checkpoint HTMUON
```bash
conda activate /data/project/le-lab/conda_env/htmuon
export HF_HOME=/data/project/le-lab/conda_env/.hf_cache
export HF_TOKEN=<your-hf-token>       

python /data/project/le-lab/Soren/runs/.bench/upload_ckpt.py tucnguyen \
    /data/project/le-lab/Soren/runs/htmuon_interval_1b_97887/model_15000 \
    /data/project/le-lab/Soren/runs/htmuon_interval_1b_97887/model_15001
```

## Uploading the model checkpoint AdamW
```bash
conda activate /data/project/le-lab/conda_env/htmuon
export HF_HOME=/data/project/le-lab/conda_env/.hf_cache
export HF_TOKEN=<your-hf-token>       
export HF_REPO=soren-1b

python /data/project/le-lab/Soren/runs/.bench/upload_ckpt.py tucnguyen \
    /data/project/le-lab/Soren/runs/soren_1b_muonlr_0.01_98064/model_15000 \
    /data/project/le-lab/Soren/runs/soren_1b_muonlr_0.01_98064/model_15001
```

## Uploading MUON 1B checkpoint
```bash
conda activate /data/project/le-lab/conda_env/htmuon
export HF_HOME=/data/project/le-lab/conda_env/.hf_cache
export HF_TOKEN=<your-hf-token>       
export HF_REPO=muon-1b

python /data/project/le-lab/Soren/runs/.bench/upload_ckpt.py tucnguyen \
    /data/project/le-lab/Soren/runs/muon_1b_98877/model_15000 \
    /data/project/le-lab/Soren/runs/muon_1b_98877/model_15001
```
