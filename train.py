import tensorflow as tf
from tensorflow import keras  # type:ignore

new_unet = tf.keras.models.load_model(  # type:ignore
    "ultrasound_segmentation/output/unet_model.keras"
)
print(new_unet.summary())
loss, acc = new_unet.evaluate()
