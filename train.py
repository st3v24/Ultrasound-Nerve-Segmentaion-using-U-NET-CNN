import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import (  # type:ignore
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau,
)  # type:ignore
from tensorflow.keras.optimizers import Adam  # type:ignore
from tensorflow.keras.metrics import Recall, Precision  # type:ignore
from param.model import build_unet
import PIL as pil

train_path = os.path.join("dataset", "train")


def load_dataset(path):
    image_path = os.path.join(path, "images")
    return image_path


print(load_dataset(train_path))
