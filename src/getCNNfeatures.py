# math related
print("Importing Math packages such as numpy")
import numpy as np

np.set_printoptions(threshold=np.nan)

# tensorflow related
print("Importing tensorflow related packages")
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm

vgg_dir = 'tensorflow_vgg/'
# Make sure vgg exists
if not isdir(vgg_dir):
    raise Exception("VGG directory doesn't exist!")

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(vgg_dir + "vgg16.npy"):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='VGG16 Parameters') as pbar:
        urlretrieve(
            'https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy',
            vgg_dir + 'vgg16.npy',
            pbar.hook)
else:
    print("Parameter file already exists!")

import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils


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

        print("Running Getting Hash for " + path)

        self.feed_dict = {self.input_: self.img}
        self.hash = sess.run(self.vgg.relu6, feed_dict=self.feed_dict)

        return (self.hash[0])


hash = Find_hash()

import os

path = "data"
index = {}
with open("index", 'w') as f:
    with tf.Session() as sess:
        for i in os.listdir(path):
            x = hash.detect_hash(os.path.join("data", i),sess)
            index[i[:len(i) - 3]] = x
            f.write('%s:%s\n' % (i[:len(i) - 3], x))
