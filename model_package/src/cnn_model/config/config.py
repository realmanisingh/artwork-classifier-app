# Importing packages
import pathlib
import cnn_model

# Configuring the directories that will be needed for the package
PACKAGE_ROOT = pathlib.Path(cnn_model.__file__).resolve().parent
print(PACKAGE_ROOT)

# Image configurations
IMG_SIZE = (256, 256)
IMG_SHAPE = (256, 256, 3)


