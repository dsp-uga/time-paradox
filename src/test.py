from clustering.kmeanscluster import Kmeans
from pre_processing.dataIO import Data
from post_prediction.createOutputCSV import FormatCsv
import numpy as np
import pandas as pd
pd.options.display.max_colwidth
import sys

test_json_path = sys.argv[1]
clusters_path = sys.argv[2] 

data = Data()
formatter = FormatCsv()
json_test, test_hashes = data.loadJson(test_json_path)
kmeans = Kmeans(json_hash=json_data,hashs=hashes)
clusters = kmeans.load(clusters_path)
csvfile = open('prediction.csv','wb')
for i in range(len(test_hashes)):
    imgid = json_test[i]['id']
    clstrid = kmeans.predict(clusters,np.array([test_hashes[i]]))
    cluster_mem = kmeans.cluster_members(kmeans.labels(clusters),clstrid).tolist()
    line = str(imgid)+', '
    for j in cluster_mem:
        line = line + str(json_test[j]['id']) + ' '
    csvfile.write(line)
    csvfile.write('\n')
csvfile.close()
formatter.formatcsv('test.csv', 'prediction.csv', 'submission.csv')
print("Submission csv ready")




 
    
