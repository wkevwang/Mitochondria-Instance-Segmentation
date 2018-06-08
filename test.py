import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from train import MitocondriaConfig

class InferenceConfig(MitocondriaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", 
                        config=inference_config,
                        model_dir=MODEL_DIR)
model_path = model.find_last()[1]
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
results = []
for image_id in dataset_val.image_ids:
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                        image_id, use_mini_mask=False)
    predictions = model.detect([original_image], verbose=1)
    results.append(predictions[0])
np.save('results.npy', results)

originals = np.load('originals.npy')
results = np.load('results.npy')

for i, r in enumerate(results):
    print('Image', str(i + 1))
    for m in range(r['masks'].shape[2]):
        pixels = 0
        mask = r['masks'][:,:,m]
        for row in mask:
            for elem in row:
                if elem == 1:
                    pixels += 1
        area = int((pixels * 22.25 / 1000000) * 1000) / 1000 # in square micrometers
        print('Mitochondria', str(m), '| Area:', str(area) + 'um')
    print()