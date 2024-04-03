import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from glob import glob
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
from param.metrics import dice_loss, dice_coef, iou
import PIL as pil

parent_path = "ultrasound_segmentation"
dataset_path = os.path.join(parent_path, "dataset")
train_path = os.path.join(dataset_path, "train")
output_path = os.path.join(parent_path, "output")
H, W = 512, 512


def numeric_sort(filename):
    return int(filename.split("_")[1].split(".")[0])


def load_dataset(path, split=0.1):
    image_path = os.path.join(path, "images")
    mask_path = os.path.join(path, "masks")
    images = os.listdir(image_path)
    masks = os.listdir(mask_path)
    sorted_images = sorted(images, key=numeric_sort)
    sorted_masks = sorted(masks, key=numeric_sort)
    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(
        sorted_images, test_size=split_size, random_state=42
    )
    train_y, valid_y = train_test_split(
        sorted_masks, test_size=split_size, random_state=42
    )

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0  # type:ignore
    x = x.astype(np.float32)
    return x


def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x / np.max(x)  # type:ignore
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])  # type:ignore
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(X, Y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(4)
    return dataset


if __name__ == "__main__":
    """Seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)

    batch_size = 2
    lr = 1e-5
    num_epochs = 10
    model_path = os.path.join(output_path, "model.keras")
    csv_path = os.path.join(output_path, "data.csv")

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(train_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    # """ Model """
    # model = build_unet((H, W, 3))
    # metrics = [dice_coef, iou, Recall(), Precision()]
    # model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)

    # callbacks = [
    #     ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    #     ReduceLROnPlateau(
    #         monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7, verbose=1
    #     ),
    #     CSVLogger(csv_path),
    # ]

    # model.fit(
    #     train_dataset,
    #     epochs=num_epochs,
    #     validation_data=valid_dataset,
    #     callbacks=callbacks,
    # )
