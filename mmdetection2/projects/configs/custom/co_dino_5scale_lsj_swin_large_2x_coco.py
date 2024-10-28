# co_dino_5scale_lsj_swin_large_1x_coco.py의 스케쥴x2, 12 -> 24로 확장
_base_ = [
    '/data/ephemeral/home/Co-DETR/projects/configs/custom/co_dino_5scale_lsj_swin_large_1x_coco.py'
]
# 모델 설정
model = dict(
    backbone=dict(drop_path_rate=0.5))

lr_config = dict(policy='step', step=[20]) # 20 에폭에서 학습률 감소

runner = dict(type='EpochBasedRunner', max_epochs=24) # 24 에폭
