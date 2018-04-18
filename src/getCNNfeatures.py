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


class Find_hash(object):
    def __init__(self):
        # The graph ready
        print("Reconstructing Network")
        self.input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.vgg = vgg16.Vgg16()
        self.vgg.build(self.input_)

    def detect_hash(self, path,sess):
        self.img = utils.load_image(path)
        self.img = self.img.reshape((1, 224, 224, 3))

        print("Getting Hash for "+path)

        self.feed_dict = {self.input_: self.img}
        self.code = sess.run(self.vgg.relu6, feed_dict=self.feed_dict)

        return (self.code[0])


hash = Find_hash()

import os

path = "data"
index = {}
op = open("index.json", "w")    # writing hashes to json file
op.write("[\n")

imgs = os.listdir(path)         # list of images
imcount = len(imgs)             # no. of images
with tf.Session() as sess:
    for i in range(imcount):        # converting to json format {id: _, hash: _}
        _id = imgs[i][:len(imgs[i]) - 4]    # image id
        _hash = hash.detect_hash(os.path.join(path, imgs[i]),sess)   # vgg16 hash
        _hash = _hash.tolist()
        index[_id] = _hash

        # writing to json
        op.write('{\n')
        op.write('\t"id": "%s",\n' % _id)
        op.write('\t"hash": [')

        for h in _hash[:-1]:
            op.write('%.4f,' % h)
        op.write('%.4f]\n' % _hash[-1]) # last element of hash (should avoid ',')

        if i != imcount - 1:
            op.write('},\n')
        else:
            op.write('}\n')         # last entry to json (should avoid ',')

op.write("]")
op.close()
