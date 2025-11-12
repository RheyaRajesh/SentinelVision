import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import yaml
from pathlib import Path

# Ensure output dirs
os.makedirs('data_yolo/train/images', exist_ok=True)
os.makedirs('data_yolo/train/labels', exist_ok=True)
os.makedirs('data_yolo/val/images', exist_ok=True)
os.makedirs('data_yolo/val/labels', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

CLASSES = ['fire', 'smoke', 'fog', 'helmet', 'no_helmet', 'door_open', 'door_closed', 'mask', 'weapon', 'violence', 'abandoned', 'loitering']
class_to_id = {cls: i for i, cls in enumerate(CLASSES)}

def perform_eda(data_dir='data'):
    """Exploratory Data Analysis."""
    files = []
    labels = []
    for root, dirs, _ in os.walk(data_dir):
        for file in (Path(root) / f for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))):
            if file.exists():
                files.append(str(file))
                label = os.path.basename(root)
                labels.append(label)
    
    df = pd.DataFrame({'file': files, 'label': labels})
    print("Label Distribution:\n", df['label'].value_counts())
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df['label'].value_counts().plot(kind='bar', ax=ax, color='blue')
    ax.set_title('Dataset Distribution')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/eda_bar.png')
    plt.close()
    
    # Image sizes
    sizes = []
    for f in files[:100]:  # Sample to avoid long run
        img = cv2.imread(f)
        if img is not None:
            sizes.append(img.shape[:2])
    df_sizes = pd.DataFrame(sizes, columns=['height', 'width'])
    print("\nImage Sizes Stats:\n", df_sizes.describe())
    df_sizes.to_csv('outputs/sizes_stats.csv')

def prepare_yolo_dataset(data_dir='data', split_ratio=0.8):
    """Convert to YOLO format: images/labels split."""
    all_files = []
    all_labels = []
    for root, dirs, _ in os.walk(data_dir):
        label = os.path.basename(root)
        if label in CLASSES:  # Only process known classes
            for f in os.listdir(root):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append(os.path.join(root, f))
                    all_labels.append(label)
    
    # NO STRATIFY — fixes your error!
    files_train, files_val = train_test_split(
        all_files, test_size=1-split_ratio, random_state=42  # Removed stratify & labels
    )
    # Assign labels based on file path
    labels_train = [os.path.basename(os.path.dirname(f)) for f in files_train]
    labels_val = [os.path.basename(os.path.dirname(f)) for f in files_val]
    
    def copy_and_label(files, labels, out_img_dir, out_label_dir):
        for file, label in zip(files, labels):
            img_name = Path(file).name
            label_name = Path(file).stem + '.txt'
            
            # Copy image
            img = cv2.imread(file)
            if img is not None:
                cv2.imwrite(os.path.join(out_img_dir, img_name), img)
                
                # Create label: class_id center_x center_y width height (normalized, full image)
                cls_id = class_to_id.get(label, 0)  # Default to 0 if unknown
                label_line = f"{cls_id} 0.5 0.5 1.0 1.0\n"
                with open(os.path.join(out_label_dir, label_name), 'w') as lf:
                    lf.write(label_line)
    
    copy_and_label(files_train, labels_train, 'data_yolo/train/images', 'data_yolo/train/labels')
    copy_and_label(files_val, labels_val, 'data_yolo/val/images', 'data_yolo/val/labels')
    
    # Generate data.yaml
    yaml_data = {
        'train': '../data_yolo/train/images',  # Relative path
        'val': '../data_yolo/val/images',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    with open('data_yolo/data.yaml', 'w') as f:
        yaml.dump(yaml_data, f)
    print("YOLO dataset prepared.")

if __name__ == "__main__":
    perform_eda()
    prepare_yolo_dataset()
    print("Preprocessing complete. EDA plot saved to outputs/eda_bar.png")