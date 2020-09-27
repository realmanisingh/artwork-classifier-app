import cnn_model.pipeline as pipeline

import os
print(os.getcwd())

@pytest.mark.parametrize('data_path', [
    ("../cnn_model/data/test")
])
def test_resize_is_correct_shape():
    