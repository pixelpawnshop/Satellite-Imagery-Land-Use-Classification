"""
preprocess_eurosat.py
Preprocess EuroSAT images: normalize, augment, split into train/val/test sets.
"""
import os
import glob
import random
from PIL import Image
import numpy as np
import shutil
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/2750')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data/splits')
IMG_SIZE = (64, 64)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

AUGMENT = False  # Set True to enable basic augmentation

random.seed(SEED)
np.random.seed(SEED)

# Helper: Augmentation (flip, rotate)
def augment_image(img):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.5:
        img = img.rotate(90)
    return img

# Prepare output folders
def prepare_dirs():
    for split in ['train', 'val', 'test']:
        for class_name in os.listdir(DATA_DIR):
            split_dir = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(split_dir, exist_ok=True)

def get_image_paths():
    image_paths = []
    for class_name in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.isdir(class_dir):
            for img_file in glob.glob(os.path.join(class_dir, '*.jpg')):
                image_paths.append((img_file, class_name))
    return image_paths

def split_data(image_paths):
    random.shuffle(image_paths)
    n = len(image_paths)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    train = image_paths[:train_end]
    val = image_paths[train_end:val_end]
    test = image_paths[val_end:]
    return train, val, test

def process_and_save(image_list, split):
    for img_path, class_name in tqdm(image_list, desc=f'Processing {split}'):
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        if AUGMENT and split == 'train':
            img = augment_image(img)
        img = np.array(img) / 255.0  # Normalize to [0,1]
        out_dir = os.path.join(OUTPUT_DIR, split, class_name)
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        # Save as normalized jpg (for simplicity)
        img_save = Image.fromarray((img * 255).astype(np.uint8))
        img_save.save(out_path)

def main():
    prepare_dirs()
    image_paths = get_image_paths()
    train, val, test = split_data(image_paths)
    process_and_save(train, 'train')
    process_and_save(val, 'val')
    process_and_save(test, 'test')
    print('Preprocessing complete. Splits saved in data/splits/')

if __name__ == '__main__':
    main()
