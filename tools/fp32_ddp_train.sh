"""
需要跑不同的模型只需要修改一下--model, 可以改为:
    vit_tiny_patch16_224,
    vit_small_patch16_224,
    vit_small_patch32_224,
    vit_base_patch16_224,
    vit_base_patch32_224,
    vit_large_patch16_224,
    vit_large_patch32_224
"""

# 8卡ddp
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--model vit_tiny_patch16_224 --data-path /dataset/extract --batch-size 32 --output ./output --tag 8GPU_ddp --amp-opt-level O0

# 4卡ddp
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py \
--model vit_tiny_patch16_224 --data-path /dataset/extract --batch-size 32 --output ./output --tag 4GPU_ddp --amp-opt-level O0

# 2卡ddp
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py \
--model vit_tiny_patch16_224 --data-path /dataset/extract --batch-size 32 --output ./output --tag 2GPU_ddp --amp-opt-level O0