ngpu=1
tag=train_backbone_pytorch_dist_scratch

python -m torch.distributed.launch --nproc_per_node=${ngpu} cfg_train.py \
    --tcp_port 12345 \
    --batch_size ${ngpu} \
    --ckpt_name 'polarOffSet_backbone_scratch.pth' \
    --config cfgs/release/backbone.yaml \
    --tag ${tag} \
    --launcher pytorch