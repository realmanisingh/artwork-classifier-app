from cnn_architecture import model
from pipeline import Pipeline

def make_prediction(*, input_image):
    """
    Making a prediction using the inference pipeline
    param input_image: A path to a jpeg image, string
    return: A numpy array that contains a single prediction
    """
    
    pipeline = Pipeline(model)
    resized_image = pipeline.resize_image(input_image)
    prediction = pipeline.make_prediction(resized_image)
    
    return prediction




test = make_prediction(input_image="../../../data/train/c0/img_34.jpg")
print(test)