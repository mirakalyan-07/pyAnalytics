# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 19:42:59 2021

@author: Mira priyadarshini
"""
#python : Topic : ......

#standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import seaborn as sns

data = {'x': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50,57,59,52,65, 47,49,48,35,33,44,45,38,43,51,46],'y': [79,51,53,78,59,74,73,57,69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14,12,20,5,29,27,8,7]       }
data  
df = pd.DataFrame(data,columns=['x','y'])
df.head()
df.mean(),df.max(),df.min()
df.shape
print (df)

#%%kmeans clustering
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3).fit(df)
dir(kmeans)
kmeans.n_clusters
centroids=kmeans.cluster_centers_
print(centroids)
kmeans.labels_ #identify which data are grouped together
df
#run these 3 lines together
plt.scatter(df['x'],df['y'],c=kmeans.labels_.astype(float),s=200,alpha=0.5)
plt.scatter(centroids[:, 0],centroids[:,1],c='red',s=100,marker='D')
plt.show()

#%%4 clusters
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4).fit(df)
dir(kmeans)
kmeans.n_clusters
centroids=kmeans.cluster_centers_
print(centroids)
kmeans.labels_ #identify which data are grouped together
df
#run these 3 lines together
plt.scatter(df['x'],df['y'],c=kmeans.labels_.astype(float),s=200,alpha=0.5)
plt.scatter(centroids[:, 0],centroids[:,1],c='red',s=100,marker='D')
plt.show()

#%%mtcars
from pydataset import data
mtcars = data('mtcars')
mtcarsData = mtcars.copy()
id(mtcarsData)
mtcarsData.head()


kmeans=KMeans(n_clusters=3).fit(mtcarsData)

kmeans.n_clusters
kmeans.inertia_
centroids=kmeans.cluster_centers_
print(centroids)
kmeans.labels_ #identify which data are grouped together
mtcarsData.groupby(kmeans.labels_).aggregate({'mpg' :[np.mean,'count'],'wt':np.mean})

#%%need for scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
mtcarsScaledData=scaler.fit_transform(mtcarsData)
mtcarsScaledData[:5]
np.min(mtcarsScaledData[:1]),np.max(mtcarsScaledData[:1])
kmeans = KMeans( init = 'random', n_clusters=3, n_init=3, max_iter=300, random_state=42)
kmeans
kmeans.fit(mtcarsScaledData)
kmeans.inertia_
kmeans.cluster_centers_  #average or rep values
kmeans.n_iter_  #in 6 times, clusters stabilised
kmeans.labels_[:5]
kmeans.cluster_centers_.shape
kmeans.cluster_centers_[0:2]
mtcarsData.groupby(kmeans.labels_).mean()
clustersKM =kmeans.labels_
clustersKM
type(clustersKM)
mtcarsData.groupby([clustersKM])['mpg','wt'].mean()

#%%dendogram
from scipy.cluster.hierarchy import dendrogram , linkage
#Linkage Matrix
Z = linkage(mtcarsScaledData, method = 'ward')
dist = DistanceMetric.get_metric('euclidean')
dist 
dist.pairwise(mtcarsScaledData)
dendro = dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distance')
plt.show()
