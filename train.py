import tensorflow as tf
from tensorflow import keras

new_unet = tf.keras.models.load_model("ultrasound_segmentation/output/unet_model.keras")
print(new_unet.summary())
