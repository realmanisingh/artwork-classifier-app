import sys
# Giving the file access to modules in parent directories
sys.path.append("..")
from cnn_layers import inputs
from cnn_layers import base_model
from processing.data_management import image_batch

def test_inputs_is_correct_shape():
    # When
    correct_shape = [None, 256, 256, 3]
    
    # Then
    assert inputs.shape.as_list() == correct_shape

def test_feature_extractor_is_correct_shape():
    # When
    correct_shape = (64, 8, 8, 2048)
    feature_batch = base_model(image_batch)
    
    # Then
    assert feature_batch.shape == correct_shape
    
    