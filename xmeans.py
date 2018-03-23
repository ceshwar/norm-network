# coding: utf-8


import pandas as pd
from pyclustering.cluster import xmeans
from pyclustering.cluster import center_initializer




df=pd.read_csv('all-prediction-matrix.csv')

X = df.T.values


column_names = [c[11:] for c in df.columns] #strip column prefix
sample=[list(i) for i in X[:100]]  #only look at the first 100 subreddits


# create object of X-Means algorithm that uses CCORE for processing
# initial centers - optional parameter, if it is None, then random centers will be used by the algorithm.
# let's avoid random initial centers and initialize them using K-Means++ method:
initial_centers = center_initializer.kmeans_plusplus_initializer(sample, 2).initialize();
xmeans_instance = xmeans.xmeans(sample, initial_centers, ccore = True, kmax=300);

# run cluster analysis
xmeans_instance.process();

# obtain results of clustering
clusters = xmeans_instance.get_clusters();


for c in clusters:
    print [column_names[ci] for ci in c]