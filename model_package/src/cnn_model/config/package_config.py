import pathlib
import sys


# Configuring the directories that will be needed for the package
TRAINED_WEIGHTS = pathlib.Path("../../../model/trained_weights/cp.ckpt")
DATA_DIR = pathlib.Path("../../../data/train/")

# Image configurations
IMG_SIZE = (256, 256)
IMG_SHAPE = (256, 256, 3)

# Training configurations
LEARNING_RATE = 0.0001
BATCH_SIZE = 64


