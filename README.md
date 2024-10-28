# Object Detection for classification of recycled items

## **📘**Overview

2024.10.02 ~ 2024.10.24

This project focuses on detecting objects in recycling trash images as part of a private competition organized by Naver Connect Foundation and Upstage.


## **📘**Contributors

|김기수|문채원|안주형|은의찬|이재훈|장지우
|:----:|:----:|:----:|:----:|:----:|:----:|
| [<img src="https://github.com/user-attachments/assets/366fc4d1-3716-4214-a6ef-87f0a4c6147f" alt="" style="width:100px;100px;">](https://github.com/Bbuterfly) <br/> | [<img src="https://github.com/user-attachments/assets/ea61c11c-c577-45bb-ae8e-64dffa192402" alt="" style="width:100px;100px;">](https://github.com/mooniswan) <br/> | [<img src="https://github.com/user-attachments/assets/6bc5913f-6e59-4aae-9433-3db2c7251978" alt="" style="width:100px;100px;">](https://github.com/Ahn-latte) <br/> | [<img src="https://github.com/user-attachments/assets/22d440d4-516b-4973-a2fe-06adc145fa01" alt="" style="width:100px;100px;">](https://github.com/0522chan) <br/> | [<img src="https://github.com/user-attachments/assets/3ed91d99-0ad0-43ee-bb11-0aefc61a0a0e" alt="" style="width:100px;100px;">](https://github.com/syous154) <br/> | [<img src="https://github.com/user-attachments/assets/04f5faa7-05c4-4ecc-87f1-0befb53da70d" alt="" style="width:100px;100px;">](https://github.com/zangzoo) <br/> |

## **📘**Metrics

- mAP50

![https://user-images.githubusercontent.com/64190071/164357745-4d03deb3-6104-4706-a890-3d002a904067.png](https://user-images.githubusercontent.com/64190071/164357745-4d03deb3-6104-4706-a890-3d002a904067.png)

![https://user-images.githubusercontent.com/64190071/164357754-718a8628-872e-4f1e-9d12-4e212b2444ab.png](https://user-images.githubusercontent.com/64190071/164357754-718a8628-872e-4f1e-9d12-4e212b2444ab.png)

![https://user-images.githubusercontent.com/64190071/164357763-9d7c667a-2c5a-4b92-b4ae-6c32be0b7d34.png](https://user-images.githubusercontent.com/64190071/164357763-9d7c667a-2c5a-4b92-b4ae-6c32be0b7d34.png)

## **📰**Tools

- github
- notion
- slack
- wandb

## **📰**Folder Structure

```
├── baseline
│   ├── mmdetection2
│   │   ├── configs
│   │   ├── inference_wbf.py
│   │   ├── train.py
│   │   └── etc
│   ├── mmdetection3
│   │   └── config files
│   ├── Ultralytics
│   │   ├── RT_DETR.ipynb
│   │   ├── RT_DETR_WBF_infernece.ipynb
│   │   ├── Yolov11.ipynb
│   │   └── Yolov11_WBF_infernece.ipynb
├── dataset
│   ├── test
│   ├── test.json
│   ├── train
│   └── train.json
├── data_split.py
├── eda.ipynb
├── visualization_data.ipynb
├── wbf.py
└── sample_submission
    ├── faster_rcnn_mmdetection_submission.csv
    ├── faster_rcnn_torchvision_submission.csv
    ├── submission_ensemble.csv
    └── train_sample.csv
```

- images : 9754
    - train : 4883
    - test : 4871
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- image size :  (1024, 1024)

## **📰**Models

- Cascade RCNN
- YOLOv11
- Co-DINO
- RT-DETR

## **📰**Backbones

- Swin Transformer
- Resnet

## **📰Experiments**
![image](https://github.com/user-attachments/assets/aa7fe374-df98-4a97-b3e8-d80ae2e57b71)

![train_batch9921_41_46441c1a381ad986227e](https://github.com/user-attachments/assets/e2535cce-6b17-4713-a822-7f906d6e0a18)


| Exp | mAP |
| --- | --- |
| Yolov11(5), RT-DETR(5), CO-DINO(2) | 0.6760 |
| Co-dino_r50(2), Co-dino_swin(5) | 0.6590 |
| Co-dino_swin(5),Co-dino_r50(2),RT-DETR(5) | 0.6797 |
| Co_dino_swin(5), RT-DETR(5), Yolov11(5) | 0.6834 |


