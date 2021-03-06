import pathlib
import cnn_model

# Configuring the directories that will be needed for the package
PACKAGE_ROOT = pathlib.Path(cnn_model.__file__).resolve().parent
TRAINED_WEIGHTS = PACKAGE_ROOT / "trained_weights/cp.ckpt"
TRAIN_DATA = PACKAGE_ROOT / "data/train"
TEST_DATA = PACKAGE_ROOT / "data/test"

# Image configurations
IMG_SIZE = (256, 256)
IMG_SHAPE = (256, 256, 3)

# Training configurations
LEARNING_RATE = 0.0001
BATCH_SIZE = 64


