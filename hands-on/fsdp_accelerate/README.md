# Llama 3.2 1B FSDP Training Guide

This repository contains the setup and scripts for training **Llama-3.2-1B** using PyTorch FSDP (Fully Sharded Data Parallel), Hugging Face TRL, and Accelerate on an HPC cluster using SLURM.

## 1. Environment Setup (Critical)

Due to compiler mismatches on the cluster (CUDA 12.6 driver vs. available compiler modules), we **must** use specific versions of PyTorch and pre-compiled wheels for Flash Attention. **Do not install the default `pip install flash-attn` as it will probably fail to compile.**

### Step 1: Create and Activate Virtual Environment
Run this on the login node:
```bash
# Create venv
python -m venv venv

# Activate venv
source venv/bin/activate
```

### Step 2: Install PyTorch (Stable 2.4.1)
We use PyTorch 2.4.1 with CUDA 12.1 compatibility, which works seamlessly with the cluster's CUDA 12.6 driver.
```bash
pip install torch==2.4.1 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### Step 3: Install Flash Attention (Pre-compiled Wheel)
We install a binary wheel to avoid the `nvcc` architecture mismatch errors (specifically the `compute_120` error).
```bash
pip install [https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl)
```

### Step 4: Install Remaining Dependencies
Install the following:

**requirements.txt**
```text
ninja
packaging
transformers
datasets
trl
accelerate
liger-kernel
tensorboard
huggingface_hub
```

**Install command:**
```bash
pip install -r requirements.txt
```

---

## 2. Configuration Files

Ensure the following configuration files are present in your directory.

### `fsdp_config.yaml`
This configures Accelerate to use FSDP with BF16 precision.

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
```

---

## 3. Required Code Modifications

By default, the training script has optimizations disabled. To ensure the job finishes within the time limit and fits in memory, you **must** edit `train_1b_llama_fsdp.py`.

### A. Enable Flash Attention 2
Locate the `load_model_and_tokenizer` function and uncomment the implementation line:

```python
# In train_1b_llama_fsdp.py

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    attn_implementation="flash_attention_2", # <--- UNCOMMENT THIS
    dtype=torch.bfloat16
)
```

### B. Enable Liger Kernels
Locate the `main` function. Inside the `SFTConfig` initialization, change `use_liger_kernel` to `True`.

```python
# In train_1b_llama_fsdp.py

sft_config = SFTConfig(
    # ... other args ...
    use_liger_kernel=True, # <--- CHANGE TO TRUE
    # ...
)
```

---

## 4. Job Submission (SLURM)

We use a SLURM script to load the correct modules (CUDA 12.6, NCCL, cuDNN) and launch the job.

1.  **Authenticate with Hugging Face** (Llama 3.2 is a gated model):
    ```bash
    huggingface-cli login
    # Paste your token when prompted
    ```

2.  **Create `submit_job.sh`:**

```bash
#!/bin/bash
#SBATCH --job-name=fsdp-llama-1b
#SBATCH --partition=gpu_a100
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=72

# Load modules
module load 2024
module load cuDNN/9.5.0.50-CUDA-12.6.0 
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

# Set project/data directory
export TEACHER_DIR=/projects/0/jhssrf023

# Set PYTHONPATH
export PYTHONPATH=${TEACHER_DIR}/JHS_installations/Python/lib/python3.11/site-packages:$PYTHONPATH

# Set the Hugging Face cache directory
# Ensure this directory exists or you have write permissions
export HF_HOME=${TEACHER_DIR}/JHS_cache/huggingface

source ./venv/bin/activate

echo "Starting accelerate launch..."
echo "Hugging Face cache is set to: $HF_HOME"

# Run the training script
accelerate launch --config_file fsdp_config.yaml train_1b_llama_fsdp.py \
    --output_dir "my-new-llama-model"

echo "Job finished."
```

3.  **Submit the job:**
    ```bash
    sbatch submit_job.sh
    ```
