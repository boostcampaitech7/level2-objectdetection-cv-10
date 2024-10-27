import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
import argparse
import os

IMG_WIDTH = 1024
IMG_HEIGHT = 1024

def parse_args():
    parser = argparse.ArgumentParser(description= 'Weighted Box Fusion using cocodataset csv files')
    parser.add_argument('-f', '--files', nargs='+', required=True)
    parser.add_argument('--work-dir', default='none')
    parser.add_argument('--output-file', default='none')
    parser.add_argument('--iou', default=0.5)
    parser.add_argument('--skip', default=0.05)

    args = parser.parse_args()

    return args

def apply_wbf_to_image(image_predictions, iou_thr=0.5, skip_box_thr=0.05):
    boxes_list = image_predictions['boxes']
    scores_list = image_predictions['scores']
    labels_list = image_predictions['labels']

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )

    boxes = [
        [x_min * IMG_WIDTH, y_min * IMG_HEIGHT, x_max * IMG_WIDTH, y_max * IMG_HEIGHT]
        for (x_min, y_min, x_max, y_max) in boxes
    ]

    return boxes, scores, labels

def main():
    args = parse_args()

    if args.work_dir != 'none':
        work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = '.'
    
    output_file_name = 'output.csv'
    if args.output_file != 'none':
        output_file_name = args.output_file
        
    all_predctions = []

    for file in args.files:
        df = pd.read_csv(file)
        all_predctions.append(df)

    merged_df = pd.concat(all_predctions, ignore_index=True)
    predictions_by_image = {}

    for _, row in merged_df.iterrows():
        image_id = row['image_id']
        prediction_string = row['PredictionString']
        if type(prediction_string) == float:
            continue
        predictions = prediction_string.split()

        boxes, scores, labels = [], [], []

        # PredictionString 파싱: 클래스, 스코어, 박스 좌표 추출
        for i in range(0, len(predictions), 6):
            cls = int(predictions[i])
            score = float(predictions[i + 1])
            x_min = float(predictions[i + 2]) / IMG_WIDTH  # 정규화
            y_min = float(predictions[i + 3]) / IMG_HEIGHT  # 정규화
            x_max = float(predictions[i + 4]) / IMG_WIDTH  # 정규화
            y_max = float(predictions[i + 5]) / IMG_HEIGHT  # 정규화

            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(score)
            labels.append(cls)

        if image_id not in predictions_by_image:
            predictions_by_image[image_id] = {'boxes': [], 'scores': [], 'labels': []}

        predictions_by_image[image_id]['boxes'].append(boxes)
        predictions_by_image[image_id]['scores'].append(scores)
        predictions_by_image[image_id]['labels'].append(labels)

        wbf_results = []

    for image_id, predictions in predictions_by_image.items():
        boxes, scores, labels = apply_wbf_to_image(predictions, iou_thr=float(args.iou), skip_box_thr=float(args.skip))

        # WBF 결과를 PredictionString 형식으로 변환
        prediction_string = ' '.join([
            f"{int(label)} {score:.6f} {x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f}"
            for (x_min, y_min, x_max, y_max), score, label in zip(boxes, scores, labels)
        ])

        wbf_results.append({'PredictionString': prediction_string, 'image_id': image_id})

    # 8. 결과를 새로운 CSV 파일로 저장
    wbf_df = pd.DataFrame(wbf_results)

    try:
        wbf_df.to_csv(f'{work_dir}/{output_file_name}.csv', index=None)
        print(f'save {output_file_name} at {work_dir}/')
    except Exception as e:
        print('Error. Please Check args')

if __name__ == '__main__':
    main()