import cnn_model.config.package_config as config
from .cnn_layers import inputs, preprocess_input, base_model, global_average, dropout, prediction
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop

# Creating the CNN model architecture
inputs = inputs
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average(x)
x = dropout(x)
outputs = prediction(x)
model = Model(inputs, outputs)

# Loading the saved weights from training to the untrained model
model.load_weights(config.TRAINED_WEIGHTS)

# Compiling the model so that it can be used for inference
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = RMSprop(lr=config.LEARNING_RATE/10),
                  metrics=['accuracy'])

