import pandas as pd
import numpy as np

#Libraries for  Model
from sklearn.model_selection import train_test_split
from sklearn import cluster
from sklearn.cluster import KMeans

tempdf = pd.read_csv("tempdf.csv")
hdidf = pd.read_csv("hdidf.csv")
geohdidf = pd.read_csv("geohdidata.csv")


#kmeans model
kmeansdf = geohdidf[['Latitude','Longitude','Total Ecological Footprint']]
kmeansdftarget = geohdidf[['Latitude','Longitude','Total Ecological Footprint','Development Tier']]

# Using KMeans cluster to create distinct geographic clusters of Human Development based on area
# We're mapping the bottom .8 (High, Medium, Low)
kmeans = KMeans(n_clusters = 3)
kmeans.fit(kmeansdf)

# Graphing KMeans cluster

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
pred_kmeans = kmeans.fit_predict(kmeansdf)

plt.figure(figsize=(10,5))
plt.scatter(kmeansdf['Longitude'],kmeansdf['Latitude'], s = 75, c = kmeansdf['Total Ecological Footprint'])

plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],s=150,c='red')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('K-Means Cluster Analysis of Global HDI Tiers below .8')
plt.colorbar()
#1:0 #4



'''The KMeans Cluster above maps to the bottom .8 of the Development Tiers 
(low tier, medium tier, high tier) by the geographic area and the Total Ecological 
Footprint '''


