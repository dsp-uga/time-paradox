print("Importing OS related packages such as read write etc")
# OS related such as read write etc
import os


# math related
print("Importing Math packages such as numpy")
import numpy as np

np.set_printoptions(threshold=np.nan)

# tensorflow related
print("Importing tensorflow related packages")
import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

# matlab packages
print("Importing Matlab Packages")
import matplotlib.pyplot as plt

# Scikit Packages
print("Importing Scikit Packages")
from scipy.ndimage import imread
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit


class CnnHash(object):
    def __init__(self):
        # The graph ready
        print("Reconstructing Network")
        self.input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.vgg = vgg16.Vgg16()
        self.vgg.build(self.input_)

    def detect_hash(self, path,sess):
        self.img = utils.load_image(path)
        self.img = self.img.reshape((1, 224, 224, 3))

#        print("Getting Hash for "+path)

        self.feed_dict = {self.input_: self.img}
        self.code = sess.run(self.vgg.relu6, feed_dict=self.feed_dict)

        return (self.code[0])


