# 고정 코드
# ResNet50 backbone Co-DETR 파일 로드
_base_ = [
    '/data/ephemeral/home/cw/Co-DETR/projects/configs/custom/co_dino_5scale_lsj_r50_1x_coco.py'
]
# pretrained Swin Transformer 
pretrained = 'models/swin_large_patch4_window12_384_22k.pth'

# 모델 설정
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        out_indices=(0, 1, 2, 3),
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True, # Default: False였음
        pretrained=pretrained),
    neck=dict(in_channels=[192, 192*2, 192*4, 192*8]),
    query_head=dict(
        transformer=dict(
            encoder=dict(
                # number of layers that use checkpoint
                with_cp=6))))

# 이미지 정규화
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024, 1024) # trash dataset에 맞춰 변경

# 학습 데이터 pipeline
train_pipeline = [
     dict(type='LoadImageFromFile'),
     dict(type='LoadAnnotations', with_bbox=True),
     dict(
         type='Resize',
         img_scale=image_size,
         ratio_range=(0.1, 2.0),
         multiscale_mode='range',
         keep_ratio=True),
     dict(
         type='RandomCrop',
         crop_type=(768, 1024),
         crop_size=image_size,
         allow_negative_crop=True),
     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
     dict(type='RandomFlip', flip_ratio=0.5),
     dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
     dict(type='Normalize', **img_norm_cfg),
     dict(type='DefaultFormatBundle'),
     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# 테스트 데이터 pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# 데이터셋 설정
data_root = '/data/ephemeral/home/dataset/'
dataset_type = 'CocoDataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(filter_empty_gt=False, pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))