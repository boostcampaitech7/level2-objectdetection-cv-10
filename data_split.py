import json
import numpy as np
from collections import defaultdict
# pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
import os

def apply_multilabel_stratified_kfold(annotations_file, n_splits=5, shuffle=True, random_state=42):
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # image_id 기준으로 image_info 생성
    image_info = {img['id']: img for img in data['images']}
    
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # image_id 기준으로 각 이미지 annotation 그룹화
    image_annotations = defaultdict(list)
    for ann in data['annotations']:
        image_annotations[ann['image_id']].append(ann['category_id'])
    
    X = np.array(list(image_info.keys())) 
    y = [list(set(image_annotations[img_id])) for img_id in X] # 각 img_id에 해당하는 리스트 생성
    
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y)
    
    # MultiLabelStratifiedKFold 적용
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    fold_data = []
    
    # 각 폴드 나누고, train/val 세트 분리
    for fold, (train_index, val_index) in enumerate(mskf.split(X, y_bin)):
        train_image_ids = X[train_index]
        val_image_ids = X[val_index]
        
        # train data
        train_data = {
            'images': [image_info[img_id] for img_id in train_image_ids],
            'annotations': [ann for img_id in train_image_ids for ann in data['annotations'] if ann['image_id'] == img_id],
            'categories': data['categories']
        }
        
        # val data
        val_data = {
            'images': [image_info[img_id] for img_id in val_image_ids],
            'annotations': [ann for img_id in val_image_ids for ann in data['annotations'] if ann['image_id'] == img_id],
            'categories': data['categories']
        }
        
        fold_data.append({
            'fold': fold,
            'train': train_data,
            'val': val_data
        })
        
        print(f"Fold {fold}:")
        print(f"  Train images: {len(train_data['images'])}")
        print(f"  Validation images: {len(val_data['images'])}")
        print(f"  Train annotations: {len(train_data['annotations'])}")
        print(f"  Validation annotations: {len(val_data['annotations'])}")
        print()

    return fold_data

def save_fold_data(fold_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fold in fold_data:
        fold_num = fold['fold']
        
        # train data 저장
        train_file = os.path.join(output_dir, f'train_fold{fold_num}.json')
        with open(train_file, 'w') as f:
            json.dump(fold['train'], f)
        
        # val data 저장
        val_file = os.path.join(output_dir, f'val_fold{fold_num}.json')
        with open(val_file, 'w') as f:
            json.dump(fold['val'], f)
    
    print(f"Fold data saved in {output_dir}")

annotations_file = '/data/ephemeral/home/dataset/train.json'
output_dir = '/data/ephemeral/home/dataset/folds'

# MultiLabelStratifiedKFold 적용
fold_data = apply_multilabel_stratified_kfold(annotations_file, n_splits=5)

# 폴드 데이터 저장
save_fold_data(fold_data, output_dir)