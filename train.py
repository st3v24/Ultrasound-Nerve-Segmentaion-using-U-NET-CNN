import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
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

# from metrics import dice_loss, dice_coef, iou

# """ Global parameters """
# H = 512
# W = 512


# def create_dir(path):
#     """Create a directory."""
#     if not os.path.exists(path):
#         os.makedirs(path)


# def load_data(path, split=0.1):
#     images = sorted(glob(os.path.join(path, "images", "*.png")))
#     masks = sorted(glob(os.path.join(path, "masks", "*.png")))

#     split_size = int(len(images) * split)

#     train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
#     train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

#     train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
#     train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

#     return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


# def read_image(path):
#     x = cv2.imread(path, cv2.IMREAD_COLOR)
#     x = cv2.resize(x, (W, H))
#     x = x / 255.0
#     x = x.astype(np.float32)
#     return x


# def read_mask(path):
#     x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     x = cv2.resize(x, (W, H))
#     x = x / np.max(x)
#     x = x > 0.5
#     x = x.astype(np.float32)
#     x = np.expand_dims(x, axis=-1)
#     return x


# def tf_parse(x, y):
#     def _parse(x, y):
#         x = x.decode()
#         y = y.decode()

#         x = read_image(x)
#         y = read_mask(y)
#         return x, y

#     x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])  # type:ignore
#     x.set_shape([H, W, 3])
#     y.set_shape([H, W, 1])
#     return x, y


# def tf_dataset(X, Y, batch=8):
#     dataset = tf.data.Dataset.from_tensor_slices((X, Y))
#     dataset = dataset.shuffle(buffer_size=200)
#     dataset = dataset.map(tf_parse)
#     dataset = dataset.batch(batch)
#     dataset = dataset.prefetch(4)
#     return dataset


# if __name__ == "__main__":
#     """Seeding"""
#     np.random.seed(42)
#     tf.random.set_seed(42)

#     """ Directory for storing files """
#     create_dir("files")

#     """ Hyperparameters """
#     batch_size = 2
#     lr = 1e-5
#     num_epochs = 10
#     model_path = os.path.join("files", "model.h5")
#     csv_path = os.path.join("files", "data.csv")

#     """ Dataset """
#     dataset_path = "UNET/COVID"
#     (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

#     print(f"Train: {len(train_x)} - {len(train_y)}")
#     print(f"Valid: {len(valid_x)} - {len(valid_y)}")
#     print(f"Test: {len(test_x)} - {len(test_y)}")

#     train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
#     valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

#     """ Model """
#     model = build_unet((H, W, 3))
#     # metrics = [dice_coef, iou, Recall(), Precision()]
#     # model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)

#     callbacks = [
#         ModelCheckpoint(model_path, verbose=1, save_best_only=True),
#         ReduceLROnPlateau(
#             monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7, verbose=1
#         ),
#         CSVLogger(csv_path),
#     ]

#     model.fit(
#         train_dataset,
#         epochs=num_epochs,
#         validation_data=valid_dataset,
#         callbacks=callbacks,
#     )
