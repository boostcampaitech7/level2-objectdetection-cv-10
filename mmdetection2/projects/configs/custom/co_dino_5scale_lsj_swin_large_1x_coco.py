# 고정 코드
# ResNet50 backbone Co-DETR 파일 로드
_base_ = [
    '/data/ephemeral/home/Co-DETR/projects/configs/custom/co_dino_5scale_lsj_r50_1x_coco.py'
]
# pretrained Swin Transformer 
pretrained = '/data/ephemeral/home/Co-DETR/models/swin_large_patch4_window12_384_22k.pth'

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
        use_checkpoint=True, # Default: False
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
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    # 증강 추가 (Albumentations)
    dict(
        type='Albu',
        transforms=[
            dict(type='RandomSizedBBoxSafeCrop', height=768, width=768, erosion_rate=0.2, p=0.5), 
            dict(type='RandomRotate90', p=0.5), 
            dict(type='Affine', scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-15, 15), shear=(-10, 10), p=0.5), 
            dict(type='RandomBrightnessContrast', brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.3,
            filter_lost_elements=True
        ),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True
    ),
     dict(type='Normalize', **img_norm_cfg),
     dict(type='Pad', size=image_size),
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