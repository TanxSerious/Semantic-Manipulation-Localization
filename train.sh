 python -m  torch.distributed.run \
--nnodes 1 \
--nproc_per_node 1  train.py \
--config ./configs/config.yaml