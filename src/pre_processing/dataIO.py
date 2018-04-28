import os
import json
import numpy as np
import tensorflow as tf
from tempfile import TemporaryFile
import pandas as pd

class Data:

    '''
    loadJson() loads the hash values from the json file and creates a numpy array of the data.
    :param: path <= The path of the json file
    :return: json_hash <= (id, hash)
    :return: hashs <= np.array(hashs)
    '''
    def loadJson(self, path):
        json_hash = json.load(open(path))  # loads json file of hash values
        hashs = np.array(pd.read_json(path, orient='records')['hash'].tolist()) 
  
        return json_hash, hashs  # returns the json_hash(id,hash_value) and hash value which is np array.

    def genHash(self, hash_mech, input_path, output_path):
        path = input_path
        index = {}
        op = open(output_path, "w")  # writing hashes to json file
        op.write("[\n")
        img_hash = TemporaryFile()
        imgs = os.listdir(path)  # list of images
        imcount = len(imgs)  # no. of images
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.49) # Wil luse only 50% of GPu so that you can run multiple instance 
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            for i in range(imcount):  # converting to json format {id: _, hash: _}
                _id = imgs[i][:len(imgs[i]) - 4]  # image id
                _hash = hash_mech.detect_hash(os.path.join(path, imgs[i]), sess)  # vgg16 hash
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

                print ("hashing image",i)


        op.write("]")
        op.close()
      
            


    def txttoJson(self):
        f = open("index.txt", "r")		# input file in which hashes are stored
        op = open("hashes.json", "w")	# output json file in format {id: _, hash: _}
        op.write('[\n')

        _id = ""
        _hash = []
        imcount = 0

        for word in f.read().split():
            if ':' in word:  			# start of new hash
                imcount += 1
                if word[-1] == '[':
                    _id = word[:-3]
                else:
                    _id, h = word.split('.:[')
                    _hash += [float(h)]

            elif ']' in word:			# end of hash
                if word[0] == ']':
                    pass
                else:
                    h = float(word[:-1])
                    _hash += [h]

                # write to file after last line of each hash is read
                op.write('{\n')
                op.write('\t"id": "%s",\n' % _id)
                op.write('\t"hash": [')

                for h in _hash[:-1]:
                    op.write('%.5f,' % h)
                op.write('%.5f]\n' % _hash[-1])  # last element of hash (should avoid ',')
                op.write('},\n')

                # reset id and hash
                _id = ""
                _hash = []

                # signal for every 1000 images
                if imcount % 1000 == 0:
                    print(imcount, "images converted")

            else:						# internal hash values
                _hash += [float(word)]

        op.write(']')
        print("# of images: ", imcount)

        op.close()
        f.close()
