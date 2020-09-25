import tensorflow as tf
import cnn_model.config.package_config as config

# Input layer
inputs = tf.keras.Input(shape=config.IMG_SHAPE)

# The layer for preprocessing the input so that it can be accepted by the ResNet50V2 model
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

# The base model layer, which is the ResNet50V2 model
base_model = tf.keras.applications.ResNet50V2(
    input_shape=config.IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Layer for converting the 5x5x2048 feature blocks to a vector of size 2048
global_average = tf.keras.layers.GlobalAveragePooling2D()

# Layer for dropout regularization
dropout = tf.keras.layers.Dropout(0.2)

# Dense layer in order to get a prediction
prediction = tf.keras.layers.Dense(10)

import os
print(os.getcwd())