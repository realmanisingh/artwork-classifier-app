import sys
# Giving the file access to modules in parent directories
sys.path.append("..")
import config.package_config as config
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data.experimental import cardinality

# Creating the data loader for the training set
train_data = image_dataset_from_directory(
    directory="../../../data/train/",
    shuffle=True,
    validation_split=0.2,
    subset='training',
    seed=102,
    batch_size=config.BATCH_SIZE,
    image_size=config.IMG_SIZE
)

# Creating the data loader for the validation set
validation_data = image_dataset_from_directory(
    directory="../../../data/train/",
    shuffle=True,
    validation_split=0.2,
    subset='validation',
    seed=102,
    batch_size=config.BATCH_SIZE,
    image_size=config.IMG_SIZE
)

# Creating a test set from the validation set, 20% of the validation set batches
val_batches = cardinality(validation_data)
test_data = validation_data.take(val_batches // 5)
validation_data = validation_data.skip(val_batches // 5)
    
