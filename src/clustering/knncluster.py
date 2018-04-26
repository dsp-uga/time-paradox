from sklearn.neighbors import NearestNeighbors

class knn:
    def __init__(self, no_neighrs, algo='ball_tree'):
        self.neigbrs = no_neighrs
        self.algo = algo

    def clustering(self, vecs):
        nebrs = NearestNeighbors(n_neighbors=self.neigbrs, algorithm=self.algo).fit(vecs)
        return nebrs

    def predict(self, nebrs, t_vecs):
        distances, indices = nebrs.kneighbors(t_vecs)
        return distances, indices
        
