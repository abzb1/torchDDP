torchrun \
--nnodes=2 \
--nproc_per_node=1 \
--max_restarts=3 \
--rdzv_id=$SLURM_JOB_ID \
--rdzv_backend=c10d \
--rdzv_endpoint="dolphin" \
ddp.py
