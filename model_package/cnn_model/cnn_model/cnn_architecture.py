import config.package_config as config
import cnn_layers as layers
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop

# Creating the CNN model architecture
inputs = layers.inputs
x = layers.preprocess_input(inputs)
x = layers.base_model(x, training=False)
x = layers.global_average(x)
x = layers.dropout(x)
outputs = layers.prediction(x)
model = Model(inputs, outputs)

# Loading the saved weights from training to the untrained model
model.load_weights(config.TRAINED_WEIGHTS)

# Compiling the model so that it can be used for inference
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = RMSprop(lr=config.LEARNING_RATE/10),
                  metrics=['accuracy'])

