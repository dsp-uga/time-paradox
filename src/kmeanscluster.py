from sklearn.cluster import KMeans
import numpy as np
from sklearn.externals import joblib
import time
np.set_printoptions(threshold=np.nan)
import json

'''
loadJson() loads the hash values from the json file and creates a numpy array of the data.
:param: path <= The path of the json file
:return: json_hash <= (id, hash)
:return: hashs <= np.array(hashs)
'''
def loadJson(path):
    json_hash = json.load(open(path))#loads json file of hash values
    hashs = np.array(json_hash[0]['hash'])#intialize the feature vector data in np array
    for i in range(len(json_hash)):
        if i == 0:
           continue
        else:
           x = np.array(json_hash[i]['hash'])
           hashs = np.vstack((hashs, x))#stack the new hash vector vertically  

    return json_hash, hashs #returns the json_hash(id,hash_value) and hash value which is np array.


json_hash, hashs = loadJson("hashes.json") #get the hash values from the function.
print ("Data imported")
print ("test image id:",json_hash[10001]['id'])#Print the index of the test image.

'''
kmeans function creates the cluster of given feature vectors and return the kmeans object 
and the time taken to cluster the feature vectors.
:param: hashs <= input feature vectors.
:param: n_clust <= targeted number of clusters.
:param: inits <= number of iterations the kmeans clustering is performed.
:return: kmeans => kmeans objects consists of the clusters.
:return: (end-start) => time taken to complete the clustering.  
'''
def kmeans(hashs, n_clust, inits):
    start = time.time() #start time 
    kmeans = KMeans(n_clusters=n_clust, random_state=0, n_jobs=-2, n_init=inits).fit(hashs) #actual clustering happens
    end =  time.time()#end time
    return kmeans, (end-start) #return the clusters and time.

kmeans, time = kmeans(hashs, 1200, 30)

print ("kmeans completed\n time taken:" ,time)
joblib.dump(kmeans, 'persistence.pkl')#Saving the clustered feature vectors.

#kmeans = joblib.load('persistence.pkl')#Loading the saved clusters.
labels = np.array(kmeans.labels_)#labels for the all the feature vectors

#:TODO Change the arguments of the predict function to take the images from the test data set.
'''
predict function predicts the cluster id of a given feature vector.
:param: kmean <= clustered feature vectors(kmeans object)
:param: img_index <= index of the feature vector to be tested 
:return: cluster id of the given image feature vector.
'''
def predict(kmean, img_index):
    classid = kmean.predict(np.array([json_hash[img_index]['hash']])) #predict the cluster to which the given feature vector belongs to.
    return classid

test_img = 10001
classid = predict(kmeans, test_img)#classid

print ('prediction key:',json_hash[test_img]['id'],'\nCluster number:',classid) #print the id of the test image.
#First image of the cluster to which the test image is predited to be. 
print ("Relevent key:", json_hash[np.where(labels == classid)[0][0]]['id']) 

cluster_mem = np.where(labels == classid)[0] #members of the cluster to which the test image belongs to.
print ('Members of cluster ',cluster_mem)# Print the members of the cluster test image belongs to.


########################################################
####Sub clustering: Clustering the cluster of images####
########################################################
 
sub_clust= np.array(json_hash[cluster_mem[0]]['hash'])#intialize the sub cluster of images as numpy array.
for j in range(len(cluster_mem)):
    if j == 0:
       continue
    else:
       x = np.array(json_hash[cluster_mem[j]]['hash'])#get the feature vectors from the hash values based on the clustered members. 
       sub_clust = np.vstack((sub_clust, x))#sub cluster of images as numpy array.

sub_kmeans = KMeans(n_clusters=50, random_state=0, n_jobs=-2, n_init=30).fit(sub_clust) #Applying kmeans cluster on the cluster of images.
sub_labels = np.array(sub_kmeans.labels_)#labels of the sub clusters.
print sub_labels #print the sub labels.

sub_classid = predict(sub_kmeans, test_img)#sub cluster id of the test image.
sub_mems = np.where(sub_labels == sub_classid)[0]#members of the sub cluster of images.
print sub_mems# printing the members of the sub cluster of images.

#printing the relevent images to the given test image.
for i in range(len(sub_mems)):
    print ("Relevent keys:", json_hash[cluster_mem[sub_mems[i]]]['id'])

