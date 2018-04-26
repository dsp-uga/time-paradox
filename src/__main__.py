from pre_processing.dataIO import Data
from pre_processing.CNNfeatures import CnnHash
from clustering.kmeanscluster import Kmeans
from clustering.knncluster import knn
import sys
import numpy as np
from numpy import genfromtxt
import pandas as pd
pd.options.display.max_colwidth

hash_mech = CnnHash()  #Hasing mechanisms.
img_path = sys.argv[1] #Path for the image data
json_path = sys.argv[2] #Path for json file to be stored.(Should end in json)
test_img = sys.argv[3]
test_json_path = sys.argv[4]


data = Data()
data.genHash(hash_mech=hash_mech, input_path=img_path, output_path=json_path)


json_data, hashes = data.loadJson("hashes.json")

##########################
####K-MEANS CLUSTERING####
##########################

kmeans = Kmeans(json_hash=json_data,hashs=hashes)

clusters = kmeans.clustering(n_clust=15000,inits=30)

kmeans.save(clusters=clusters,path="clusters.pkl") #The path for saving the clusters should end in pkl

Data.genHash(hash_mech=hash_mech, input_path=test_img,output_path=test_json_path)

json_test, test_hashes = Data.loadJson(test_json_path)

clstrid = kmeans.predict(clusters,np.array([json_data[88]['hash']]))

cluster_mem = kmeans.cluster_members(kmeans.labels(clusters),clstrid)

sub_clusters = kmeans.subClustering(clusters=clusters,clusterid=clstrid,no_sub_clusts=3,sub_inits=30)

sub_cstr_id = kmeans.predict(sub_clusters,np.array([json_data[88]['hash']]))

similar_imgs = kmeans.cluster_members(labels=kmeans.labels(sub_clusters),clusterid=sub_cstr_id)


index = pd.read_csv('index.csv', sep=',',header=0)

print("Query image:",list(index.loc[index['id'] == str(json_data[88]['id'])]['url']))
 
for i in range(len(similar_imgs)):
    print ("Relevent keys:", list(index.loc[index['id'] == str(json_data[cluster_mem[similar_imgs[i]]]['id'])]['url']))

print("###########################")
print("####K-NEAREST NEIGHBOUR####")
print("###########################")

knns = knn(25)
nebrs = knns.clustering(hashes)
distns, indices = knns.predict(nebrs,np.array([json_data[88]['hash']]))

for i in indices[0]:
    print ("Relevent Keys:", list(index.loc[index['id'] == str(json_data[i]['id'])]['url']))

