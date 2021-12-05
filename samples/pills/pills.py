import os
import sys
import math
import random
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import prepare_data


class PillsConfig(Config):
    NAME = "pills"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + pill

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class PillsDataset(utils.Dataset):
    def load_pills(self, img_paths, bg_paths, white_bg_paths, count, height=128, width=128):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("pills", 1, "pill")

        num_images = len(img_paths)
        # Add images
        for i in range(count):
            img_path = img_paths[i % num_images]
            overlaid, mask, bg_path = self.random_image(height, width, img_path, bg_paths, white_bg_paths)
            self.add_image("pills", image_id=i, path=img_path,
                           width=width, height=height,
                           bg_path=bg_path, image=overlaid, mask=mask)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        return self.image_info[image_id]["image"]

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "pills":
            return info["pills"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        if info["source"] != "pills":
            return super(self.__class__, self).load_mask(image_id)

        mask = info["mask"]

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def random_image(self, height, width, img_path, bg_paths, white_bg_paths, white_bg_prob=0.3):
        # Pick random background
        if random.random() < white_bg_prob:
            bg_path = random.choice(white_bg_paths)
        else:
            bg_path = random.choice(bg_paths)
        
        img = cv2.imread(str(img_path))
        bg = cv2.imread(str(bg_path))

        # Overlay pill onto background
        overlaid, mask = prepare_data.overlay_images(img, bg)

        # Resize
        overlaid = cv2.resize(overlaid, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)

        return overlaid, mask, bg_path