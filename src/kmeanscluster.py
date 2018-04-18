from sklearn.cluster import KMeans
import numpy as np

np.set_printoptions(threshold=np.nan)
import json

json_hash = json.load(open('hashes.json'))
hashs = np.array(json_hash[0]['hash'])
for i in range(len(json_hash)):
    if i == 0:
        continue
    else:
        x = np.array(json_hash[i]['hash'])
        hashs = np.vstack((hashs, x))

print(hashs.shape)
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
print(kmeans.labels_)
kmeans.predict()

print(kmeans.cluster_centers_)
