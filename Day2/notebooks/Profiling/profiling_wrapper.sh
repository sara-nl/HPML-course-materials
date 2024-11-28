#!/bin/bash

# We have this wrapper so that we can use the SLURM_PROCID env var to create a rank-specific output filename
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -o profile_job_${SLURM_JOBID}_host_$(hostname)_rank_${SLURM_PROCID}.nsys-rep  --capture-range=cudaProfilerApi --capture-range-end=stop python mnist_classify_ddp.py --batch-size 128 --epochs 1
