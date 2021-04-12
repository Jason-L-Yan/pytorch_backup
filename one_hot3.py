import numpy as np
from numpy import argmax
from keras.utils import to_categorical


def onehot(data):
    data = np.array(data)
    encoded = to_categorical(data)

    return encoded