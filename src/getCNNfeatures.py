print("Importing OS related packages such as read write etc")
# OS related such as read write etc
import os
import csv

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


class Find_State(object):
    def __init__(self):
        # Constants
        print("Init Constants")
        self.class_size = 16
        self.no_ofcodes = 4096

        # The graph ready
        print("Reconstructing Network")
        self.input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.vgg = vgg16.Vgg16()
        self.vgg.build(self.input_)

    def detect_state(self, path):
        print("Getting image")
        # self.path="data/000c2a80838aabff.jpg"
        # self.test_imgh = imread(path)
        self.img = utils.load_image(path)
        self.img = self.img.reshape((1, 224, 224, 3))

        print("Running Network")
        with tf.Session() as sess:
            self.feed_dict = {self.input_: self.img}
            self.code = sess.run(self.vgg.relu6, feed_dict=self.feed_dict)

        return (self.code[0])


hash = Find_State()

import os

path = "data"
index = {}
with open("index", 'w') as f:
    for i in os.listdir(path):
        x = hash.detect_state(os.path.join("data", i))
        index[i[:len(i) - 3]] = x
        f.write('%s:%s\n' % (i[:len(i) - 3], x))

