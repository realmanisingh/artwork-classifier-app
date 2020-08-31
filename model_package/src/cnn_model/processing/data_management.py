import sys
# Giving the file access to modules in parent directories
sys.path.append("..")
import config.package_config as config
from tensorflow.keras.preprocessing import image_dataset_from_directory


def load_data():
    """
    Loads in the data from the data directory and creates a training set, validation set, and test set
    with an 80/20 split
    return: 3 tf.data.Dataset objects for each set of data
    """
    # Creating the data loader for the training set
    train_data = image_dataset_from_directory(
        directory=config.DATA_DIR,
        shuffle=True,
        validation_split=0.2,
        subset='training',
        seed=102,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE
    )

    # Creating the data loader for the validation set
    validation_data = image_dataset_from_directory(
        directory=config.DATA_DIR,
        shuffle=True,
        validation_split=0.2,
        subset='validation',
        seed=102,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE
    )

    # Creating a test set from the validation set, 20% of the validation set batches
    val_batches = tf.data.experimental.cardinality(validation_data)
    test_data = validation_data.take(val_batches // 5)
    validation_data = validation_data.skip(val_batches // 5)
    
    return (train_data, test_data, validation_data)
