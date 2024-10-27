import os
import gc
import torch
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.utils import get_device
from mmdet.datasets import PIPELINES
import albumentations as A
import numpy as np

from tqdm import tqdm
from mmcv.runner import HOOKS, WandbLoggerHook

# 클래스 정의
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config 파일 로드
cfg = Config.fromfile('/data/ephemeral/home/Co-DETR/projects/configs/custom/co_dino_5scale_lsj_swin_large_2x_coco.py')

# dataset 경로
root = '/data/ephemeral/home/dataset/'

# wandb custom hook
@HOOKS.register_module()
class CustomWandbLoggerHook(WandbLoggerHook):
    def __init__(self, log_evaluation_metrics=False, **kwargs):
        super(CustomWandbLoggerHook, self).__init__(**kwargs)
        self.log_evaluation_metrics = log_evaluation_metrics

    def after_val_epoch(self, runner):
        super(CustomWandbLoggerHook, self).after_val_epoch(runner)
        if self.log_evaluation_metrics and runner.log_buffer.output:
            evaluation_metrics = {key: val for key, val in runner.log_buffer.output.items()
                                  if key.startswith('val/')}
            self.wandb.log(evaluation_metrics, step=runner.epoch)

fold = 0

# 훈련 데이터셋
cfg.data.train = dict(
    type='CocoDataset',
    classes=classes,
    ann_file=os.path.join(root, f'folds/train_fold{fold}.json'),
    img_prefix=root,
    pipeline=cfg.train_pipeline
)
# 검증 데이터셋 설정
cfg.data.val = dict(
    type='CocoDataset',
    classes=classes,
    ann_file=os.path.join(root, f'folds/val_fold{fold}.json'),
    img_prefix=root,
    pipeline=cfg.test_pipeline
)
# 테스트 데이터셋
cfg.data.test = dict(
    type='CocoDataset',
    classes=classes,
    ann_file=os.path.join(root, 'test.json'),
    img_prefix=root,
    pipeline=cfg.test_pipeline
)

cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 4
cfg.data.pin_memory = True
cfg.seed = 42
cfg.gpu_ids = [0]
cfg.device = get_device()

cfg.work_dir = f'./work_dirs/co_dino_5scale_lsj_swin_large_2x_trash_fold{fold}' 
cfg.model.query_head.num_classes = 10  #출력 클래스 수

cfg.lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-5)

cfg.optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))

cfg.auto_scale_lr = dict(base_batch_size=16)

# wandb logging
cfg.log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='CustomWandbLoggerHook',
            init_kwargs=dict(
                project='Co-DETR',
                name=f'co_dino_5scale_lsj_swin_large_2x_fold{fold}',
                config={
                    'classes': classes,
                    'samples_per_gpu': cfg.data.samples_per_gpu,
                    'num_classes': cfg.model.query_head.num_classes,
                    'optimizer': cfg.optimizer,
                    'lr_config': cfg.lr_config,
                    'runner': cfg.runner,
                    'evaluation': cfg.evaluation,
                },
                tags=['co_dino', 'object_detection', f'fold_{fold}']
            ),
            log_artifact=True,
            log_evaluation_metrics=True
        )
    ]
)

# 데이터셋 with sampler, 모델 생성
datasets = [build_dataset(cfg.data.train)]
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.init_weights()

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=True)

print("학습이 끝났습니다.")

torch.cuda.empty_cache()
gc.collect()