import tensorflow as tf
import numpy as np

class Pipeline:
    def __init__(self, model):
        self.model = model
        
    def resize_image(self, image_path):
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
    
    def make_prediction(self, input_image):
        """
        Make a prediction using the saved model
        param input_image: A preprocessed image that is converted to a Tensorflow tensor
        return: A numpy array that contains a single prediction
        """
        
        return self.model.predict(input_image)
    
    def evaluate_model(self, data):
        """
        Evaluate the model using a set of images
        param data: A tf.data.Dataset object of images with labels
        return: The accuracy of the model predictions, float
        """
        
        loss, accuracy = self.model.evaluate(data)
        
        return accuracy
