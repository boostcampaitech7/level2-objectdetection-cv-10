from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# pip install ensemble_boxes
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO

def ensemble_predictions(predictions_list, iou_threshold=0.5, conf_threshold=0.05, img_size=(1024, 1024)):
    boxes_list = []
    scores_list = []
    labels_list = []

    for preds in predictions_list:
        boxes = []
        scores = []
        labels = []
        for label, objs in enumerate(preds):
            for obj in objs:
                if obj[4] > conf_threshold:
                    # 박스 좌표를 [0, 1] 범위로 정규화
                    norm_box = [
                        obj[0] / img_size[1],
                        obj[1] / img_size[0],
                        obj[2] / img_size[1],
                        obj[3] / img_size[0]
                    ]
                    # 최소 크기 설정
                    norm_box[2] = max(norm_box[2], norm_box[0] + 1e-4)
                    norm_box[3] = max(norm_box[3], norm_box[1] + 1e-4)
                    boxes.append(norm_box)
                    scores.append(obj[4])
                    labels.append(label)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_threshold,
        skip_box_thr=conf_threshold
    )

    final_predictions = []
    for i in range(len(boxes)):
        # 박스 좌표를 원래 크기로 변환
        denorm_box = [
            boxes[i][0] * img_size[1],
            boxes[i][1] * img_size[0],
            boxes[i][2] * img_size[1],
            boxes[i][3] * img_size[0]
        ]
        final_predictions.append({
            'bbox': denorm_box,
            'conf': scores[i],
            'cls': labels[i]
        })
    return final_predictions

def main():
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    cfg = Config.fromfile('/data/ephemeral/home/Co-DETR/projects/configs/custom/co_dino_5scale_lsj_swin_large_2x_coco.py')
    root = '/data/ephemeral/home/dataset/'

    if 'query_head' in cfg.model:
        if 'dn_cfg' not in cfg.model.query_head:
            cfg.model.query_head.dn_cfg = dict(type='CdnQueryGenerator')
        elif 'type' not in cfg.model.query_head.dn_cfg:
            cfg.model.query_head.dn_cfg.type = 'CdnQueryGenerator'
    
    # 테스트 데이터셋 설정
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = os.path.join(root, 'test.json')
    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 4
    cfg.seed = 42
    cfg.gpu_ids = [0]
    
    # COCO 데이터셋 로드
    coco = COCO(cfg.data.test.ann_file)
    
    # 모델 생성 및 로드
    models = []
    for fold in range(2):  # 2 fold
        cfg.model.query_head.num_classes = 10
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint_path = f'./work_dirs/co_dino_5scale_lsj_swin_large_2x_trash_fold{fold}/latest.pth'
        load_checkpoint(model, checkpoint_path, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        models.append(model)
    
    # 데이터셋 및 데이터로더 생성
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    # WBF한 추론 실행
    prediction_strings = []
    file_names = []
    
    for i, data in enumerate(tqdm(data_loader)):
        predictions_list = []
        for model in models:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            predictions_list.append(result[0])
        
        # WBF를 사용한 앙상블 예측
        ensemble_preds = ensemble_predictions(predictions_list)
        
        # 제출 형식으로 변환
        prediction_string = ' '.join([
            f"{int(pred['cls'])} {pred['conf']:.6f} "
            f"{pred['bbox'][0]:.2f} {pred['bbox'][1]:.2f} {pred['bbox'][2]:.2f} {pred['bbox'][3]:.2f}"
            for pred in ensemble_preds
        ])
        
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    
    # 최종 예측을 CSV 파일로 저장
    submission = pd.DataFrame({
        'PredictionString': prediction_strings,
        'image_id': file_names
    })
    
    output_file = './output.csv'
    submission.to_csv(output_file, index=None)
    print(f"앙상블 예측이 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()