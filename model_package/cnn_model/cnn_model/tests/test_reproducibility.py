import sys
# Giving the file access to modules in parent directories
sys.path.append("..")
from evaluation import accuracy

def test_if_test_set_accuracy_is_similar():
    assert accuracy >= 0.99 