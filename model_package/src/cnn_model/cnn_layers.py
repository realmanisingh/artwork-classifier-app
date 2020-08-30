# Importing libraries
import tensorflow as tf
import config

# The layer for preprocessing the input so that it can be accepted by the ResNet50V2 model
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

# The base model layer, which is the ResNet50V2 model
base_model = tf.keras.applications.ResNet50V2(
    input_shape=config.IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Layer for converting the 5x5x2048 feature blocks to a vector of size 2048
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Dense layer in order to get a prediction
prediction_layer = tf.keras.layers.Dense(10)