import tensorflow as tf
import numpy as np

def resize_image(*, image_path: str) -> tf.Tensor:
    """
    Resizing an image to 256x256
    param image_path: A path to the image in jpeg format
    return: An image resized to 256x256
    """
    image = tf.keras.preprocessing.image.load_img(image_path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    image_arr = np.array([image_arr])
    resized_image = tf.image.resize(image_arr, [256, 256])
    
    return resized_image

test = resize_image(image_path="../../../../data/train/c0/img_34.jpg")
print(test.shape.as_list())