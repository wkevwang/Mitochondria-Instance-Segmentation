import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

TRAINING_IMAGES_DIRECTORY = './Training/Images'
TRAINING_ANNOTATIONS_DIRECTORY = './Training/Masks'

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class MitocondriaConfig(Config):
    NAME = "mitochondria"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 816
    IMAGE_MAX_DIM = 1216
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 100
    STEPS_PER_EPOCH = 140
    VALIDATION_STEPS = 40
    
config = MitocondriaConfig()

class MitochondriaDataset(utils.Dataset):

    def __init__(self, image_names):
        super().__init__()
        self.add_class("mitochondria", 1, "mitochondria")

        for idx, img_name in enumerate(image_names):
            path = os.path.join(TRAINING_IMAGES_DIRECTORY, img_name)
            self.add_image("mitochondria", image_id=idx, path=path)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = cv2.imread(info["path"])
        return image
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        image_name_no_ext = os.path.basename(info["path"]).split('.')[0]
        masks_directory = os.path.join(TRAINING_ANNOTATIONS_DIRECTORY, image_name_no_ext)
        masks = []
        for mask_name in os.listdir(masks_directory):
            if mask_name[0] == '.':
                continue
            mask = cv2.imread(os.path.join(masks_directory, mask_name))
            mask = np.array(mask)
            mask = mask[:,:,0] > 30
            masks.append(mask.astype(np.uint8))
        masks_stack = np.dstack(masks)
        return masks_stack, np.array([1]*len(masks))
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]

training_images = []
val_images = []
for idx, image_name in enumerate(os.listdir(TRAINING_IMAGES_DIRECTORY)):
    if idx < 60:
        val_images.append(image_name)
    else:
        training_images.append(image_name)

dataset_train = MitochondriaDataset(training_images)
dataset_train.prepare()
dataset_val = MitochondriaDataset(val_images)
dataset_val.prepare()

model = modellib.MaskRCNN(mode="training", config=config,
                        model_dir=MODEL_DIR)
model.load_weights(COCO_MODEL_PATH, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                            "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=20, 
            layers="all")

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax