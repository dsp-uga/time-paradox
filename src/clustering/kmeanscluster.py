from sklearn.cluster import KMeans
import numpy as np
from sklearn.externals import joblib
import time
np.set_printoptions(threshold=np.nan)

class Kmeans():

    def __init__(self,json_hash,hashs):
        self.json_hash = json_hash
        self.hashs = hashs

    '''
    kmeans function creates the cluster of given feature vectors and return the kmeans object
    and the time taken to cluster the feature vectors.
    :param: hashs <= input feature vectors.
    :param: n_clust <= targeted number of clusters.
    :param: inits <= number of iterations the kmeans clustering is performed.
    :return: kmeans => kmeans objects consists of the clusters.
    :return: (end-start) => time taken to complete the clustering.
    '''
    def clustering(self,n_clust,inits):
        self.clusters = KMeans(n_clusters=n_clust, random_state=0, n_jobs=-2,
                               n_init=inits).fit(self.hashs) #actual clustering happens
        return self.clusters

    def save(self,clusters,path):
        joblib.dump(clusters, path)#Saving the clustered feature vectors.

    def load(self,path):
        clusters = joblib.load('persistence.pkl')#Loading the saved clusters.
        return clusters

    #:TODO Change the arguments of the predict function to take the images from the test data set.
    '''
    predict function predicts the cluster id of a given feature vector.
    :param: kmean <= clustered feature vectors(kmeans object)
    :param: img_index <= index of the feature vector to be tested 
    :return: cluster id of the given image feature vector.
    '''
    def predict(self, kmean_clust, img_vec):
        classid = kmean_clust.predict(img_vec) #predict the cluster to which the given feature vector belongs to.
        return classid

    def labels(self,clusters):
        return np.array(clusters.labels_)

    def cluster_members(self,labels,clusterid):
        return np.where(labels == clusterid)[0]

    ########################################################
    ####Sub clustering: Clustering the cluster of images####
    ########################################################

    def subClustering(self,clusters,clusterid,no_sub_clusts,sub_inits):

        labels = np.array(clusters.labels_)  # labels for the all the feature vectors
        cluster_mem = np.where(labels == clusterid)[0]  # members of the cluster to which the test image belongs to.
        sub_clust= np.array(self.json_hash[cluster_mem[0]]['hash'])#intialize the sub cluster of images as numpy array.
        for j in range(len(cluster_mem)):
            if j == 0:
               continue
            else:
               x = np.array(self.json_hash[cluster_mem[j]]['hash'])#get the feature vectors from the hash values based on the clustered members.
               sub_clust = np.vstack((sub_clust, x))#sub cluster of images as numpy array.

        sub_clusters = KMeans(n_clusters=no_sub_clusts, random_state=0, n_jobs=-2, n_init=sub_inits).fit(sub_clust) #Applying kmeans cluster on the cluster of images.
        return sub_clusters
