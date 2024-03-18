import torch
import os


DATASET_PATH = (
    "C:\\Users\\stevi\\Documents\\Python Stuff\\ultrasound_segmentation\\dataset"
)

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = "C:\\Users\\stevi\\Documents\\Python Stuff\\ultrasound_segmentation\\dataset\\train\\images"
MASK_DATASET_PATH = "C:\\Users\\stevi\\Documents\\Python Stuff\\ultrasound_segmentation\\dataset\\train\\masks"

# define the test split
TEST_SPLIT = 0.15

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes, and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64

# define the input image dimensions
INPUT_IMAGE_WIDTH = 128  # original size = 580
INPUT_IMAGE_HEIGHT = 128  # original size = 420

# define threshold to filter weak predictions
THRESHOLD = 0.5


# define the path to the output serialized model, model training plot, and testing image paths
MODEL_PATH = "C:\\Users\\stevi\\Documents\\Python Stuff\\ultrasound_segmentation\\output\\unet_us_nerve.pth"
PLOT_PATH = "C:\\Users\\stevi\\Documents\\Python Stuff\\ultrasound_segmentation\\output\\plot.png"
TEST_PATHS = "C:\\Users\\stevi\\Documents\\Python Stuff\\ultrasound_segmentation\\output\\test_paths.txt"
