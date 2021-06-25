# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 15:33:24 2021

@author: Mira priyadarshini
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df= pd.read_csv('https://raw.githubusercontent.com/DUanalytics/pyAnalytics/master/mtcars.csv')
df
index1=df.index
index1
mileage=df.mpg
mileage
time=df.qsec
time
df1 = pd.DataFrame({'mileage':mileage, 'time':time},index=index1)
df1
df2=df1.head(6)
df2
df2.plot(kind='scatter', x='mileage', y='time')
plt.scatter(df2['mileage'], df2['time'], s = 10, c = 'k')

from scipy.cluster.hierarchy import dendrogram , linkage
#Linkage Matrix
Z = linkage(df2, method = 'ward')
 
#plotting dendrogram
df2
dendro = dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distance')
plt.show()
df2

from scipy.spatial import distance
import numpy as np
#distance.euclidean([1, 0, 0], [0, 1, 0])
distance.euclidean([21.0,16.46],[21.0,17.02]) #(0,1) #closest : S1 with S2
#np.sqrt(((20-25)**2 + (25-22)**2)) #sqrt(sum(x-y)^2)

distance.euclidean([22.8,18.61],[21.4,19.44]) #(2,3)
distance.euclidean([18.7,17.02],[18.1,20.22])#(4,5)
distance.euclidean([18.1,20.22],[22.8,18.61])#(5,2)

#distance of all points in DF
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('euclidean')
dist
df.to_numpy()
dist.pai

