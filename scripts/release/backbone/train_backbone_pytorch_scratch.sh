ngpu=1
tag=train_backbone_pytorch_dist

python -m torch.distributed.launch --nproc_per_node=${ngpu} cfg_train.py \
    --tcp_port 12345 \
    --ckpt_name 'polarOffSet_backbone_scratch.pth'
    --batch_size ${ngpu} \
    --config cfgs/release/backbone.yaml \
    --tag ${tag} \
    --launcher pytorch