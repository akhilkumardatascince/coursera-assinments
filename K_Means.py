# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:24:17 2019

@author: M1050257
"""

""" Importing all required libraries """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math as mt
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.cluster import KMeans
""" --------------------------------- """

data = pd.read_csv('C:\\Santhosh\\DATA_SCIENCE\\Datasets\\Kaggle\\Admission Predict\\Admission_Predict_1.csv')
data.shape #(500,9)
list(data.columns)
"""
['Serial No.',
 'GRE Score',
 'TOEFL Score',
 'University Rating',
 'SOP',
 'LOR ',
 'CGPA',
 'Research',
 'Chance of Admit ']
"""
data.dtypes
data.info() #memory usage: 35.2 KB

data.isna().sum()
#no missing values

#upper-case all DataFrame column names
data.columns = map(str.upper, data.columns)
#data management
data_clean = data.dropna()

# subset clustering variables
cluster=data_clean[['GRE SCORE','TOEFL SCORE','UNIVERSITY RATING','SOP','LOR ','CGPA','RESEARCH']]
cluster.describe()

# standardize clustering variables to have mean=0 and sd=1
clustervar=cluster.copy()
clustervar['GRE SCORE']=preprocessing.scale(clustervar['GRE SCORE'].astype('float64'))
clustervar['TOEFL SCORE']=preprocessing.scale(clustervar['TOEFL SCORE'].astype('float64'))
clustervar['UNIVERSITY RATING']=preprocessing.scale(clustervar['UNIVERSITY RATING'].astype('float64'))
clustervar['SOP']=preprocessing.scale(clustervar['SOP'].astype('float64'))
clustervar['LOR ']=preprocessing.scale(clustervar['LOR '].astype('float64'))
clustervar['CGPA']=preprocessing.scale(clustervar['CGPA'].astype('float64'))
clustervar['RESEARCH']=preprocessing.scale(clustervar['RESEARCH'].astype('float64'))

# split data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

# Interpret 3 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()


"""
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus=pd.DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()
#OUTPUT
"""
Out[277]: 
0    133
2    131
1     86
Name: cluster, dtype: int64
"""

"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)

"""
Clustering variable means by cluster
              index  GRE SCORE    ...         CGPA  RESEARCH
cluster                           ...                       
0        231.390977   0.127419    ...     0.091507  0.280523
1        246.430233   1.145296    ...     1.220629  0.652154
2        273.908397  -0.881373    ...    -0.919047 -0.697560
"""
