#!/usr/bin/env bash
# tf oneDNN optimization
export TF_ENABLE_ONEDNN_OPTS=1
export OMP_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
# export KMP_AFFINITY=granularity=fine,none,1,0
python TextFilterService.py --num_intra_threads=18 --num_inter_threads=2