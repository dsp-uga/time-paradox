from sklearn.cluster import KMeans
import numpy as np
from sklearn.externals import joblib

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


kmeans = KMeans(n_clusters=2000, random_state=0).fit(hashs)
joblib.dump(kmeans, 'persistence.pkl')

kmeans = joblib.load('persistence.pkl')
print(kmeans.labels_)
#kmeans.predict()
#print(kmeans.cluster_centers_)
#[1 1 2 1 0]