# USAGE
# python train.py
# import the necessary packages
from param.dataset import SegmentationDataset
from param.model import UNet
from param import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np


imagePaths = []
maskPaths = []
for path in os.listdir(config.IMAGE_DATASET_PATH):
    imagePaths.append(path)
for path in os.listdir(config.MASK_DATASET_PATH):
    maskPaths.append(path)


split = train_test_split(
    imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42
)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

for i in range(0, 50):
    print(trainImages[i], ":", trainMasks[i])

print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()
