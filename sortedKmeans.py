from sklearn.cluster import KMeans 
import pandas as pd
import numpy as np

class kmeans_1d:
    
    def __init__(self, x, n = 5, binary = True):
        self.x = x
        self.date = x.index
        self.n_clusters = n
        self.kmeans = self.fit()
        self.label = self.mapping()
        self.binary = binary
    
    def fit(self):
        date = self.date
        if self.x.ndim !=2:
            self.x = np.array(self.x).reshape(-1,1)
        kmeans = KMeans(n_clusters = self.n_clusters).fit(self.x)
        return kmeans
    
    def mapping(self):
        kmeans = self.kmeans

        # map raw_label to sorted label
        label = pd.DataFrame({'center':kmeans.cluster_centers_[:,0],
                        'raw_label':kmeans.predict(kmeans.cluster_centers_[:,0].reshape(-1,1))})
        label = label.sort_values(by ='center')
        if self.binary:
            binary = np.zeros(self.n_clusters)
            binary[-1] = 1
            label['label'] = binary
        else:
            label['label'] = range(1,self.n_clusters+1)
        
        return label
    
    def predict(self, to_predict):
        kmeans = self.kmeans
        label = self.label
        date = to_predict.index
        
        if to_predict.ndim != 2:
            to_predict = np.array(to_predict).reshape(-1,1)        
        
        def label_transform(l):
            return label[label.raw_label==l]['label'].iloc[0]

        return pd.Series(np.vectorize(label_transform)(kmeans.predict(to_predict)),index = date)
    
    def train(self):
        # turn VIX return into VIX label in train set
        
        # map return to raw_label
        a = pd.DataFrame({'raw_label':self.kmeans.labels_,
                      '1fwdret': self.x[:,0]})
        
        return a.merge(self.label,on = 'raw_label',how = 'left').set_index(self.date).drop(columns = 'raw_label')
    