import os
import cv2

import numpy as np
RAW_IMAGES_DIRECTORY = '/Volumes/Purple/Data/Images'
RAW_ANNOTATIONS_DIRECTORY = '/Volumes/Purple/Data/Masks'

TRAINING_IMAGES_DIRECTORY = './Training/Images'
TRAINING_ANNOTATIONS_DIRECTORY = './Training/Masks'

RESIZE_RATIO = 4

for image_name in os.listdir(RAW_IMAGES_DIRECTORY):
    if image_name[0] == '.':
        continue
    image = cv2.imread(os.path.join(RAW_IMAGES_DIRECTORY, image_name))
    image_name_no_ext = image_name.split('.')[0]
    masks_directory = os.path.join(RAW_ANNOTATIONS_DIRECTORY, image_name_no_ext)
    for mask_name in os.listdir(masks_directory):
        if mask_name[0] == '.':
            continue
        mask = cv2.imread(os.path.join(masks_directory, mask_name))
        mask_crop = mask[0:3264, 0:4864]
        mask_resized = cv2.resize(mask_crop, (int(4864 / RESIZE_RATIO), int(3264 / RESIZE_RATIO)))
        if not os.path.exists(os.path.join(TRAINING_ANNOTATIONS_DIRECTORY, image_name.split('.')[0]))
            os.mkdir(os.path.join(TRAINING_ANNOTATIONS_DIRECTORY, image_name.split('.')[0]))
        cv2.imwrite(os.path.join(TRAINING_ANNOTATIONS_DIRECTORY, image_name.split('.')[0], mask_name.split('.')[0] + '.jpg'), mask_resized)
    image_crop = image[0:3264, 0:4864]
    image_resized = cv2.resize(image_crop, (int(4864 / RESIZE_RATIO), int(3264 / RESIZE_RATIO)))
    cv2.imwrite(os.path.join(TRAINING_IMAGES_DIRECTORY, image_name.split('.')[0] + '.jpg'), image_resized)