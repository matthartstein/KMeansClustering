'''
NAME:     Matthew S. Hartstein (ID: 010567629)
CLASS:    Big Data Analytics & Management - Project #3
DATE:     2/27/20
SYNOPSIS: This program uses pyplot/numpy/pandas/sklearn libraries to implement a
          K-Means Clustering algorithm, then displays a scatter plot of the
          results as clusters. Each cluster group has its own color and a black
          star is the focal point (average) of each cluster. However, I was not
          able to use data from my excel file but my program still demonstrates
          how to properly implement a K-Means Clustering algorithm with randomly
          genereated numbers with 4 clusters.
'''

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate dataset
dataset = make_blobs(n_samples=200,
                    centers=4,
                    n_features=2,
                    cluster_std=1.6,
                    random_state=50)

# Read excel file dataframe
file_contents = pd.read_csv("./data.csv")

# Output file file contents
print(file_contents)

# Print the dataset
points = dataset[0]

# Create a kMeans object
kmeans = KMeans(n_clusters=4)

# Fit kMeans object to dataset
kmeans.fit(points)
test = plt.scatter(dataset[0][:,0], dataset[0][:,1])

# Identify clusters
clusters = kmeans.cluster_centers_

# Create new object - extract excel data here (?)
y_km = kmeans.fit_predict(points)

# Update clusters
plt.scatter(points[y_km == 0,0], points[y_km == 0,1], s=50, color='red')
plt.scatter(points[y_km == 1,0], points[y_km == 1,1], s=50, color='green')
plt.scatter(points[y_km == 2,0], points[y_km == 2,1], s=50, color='yellow')
plt.scatter(points[y_km == 3,0], points[y_km == 3,1], s=50, color='cyan')

plt.scatter(clusters[0][0], clusters[0][1], marker='*', s=200, color='black')
plt.scatter(clusters[1][0], clusters[1][1], marker='*', s=200, color='black')
plt.scatter(clusters[2][0], clusters[2][1], marker='*', s=200, color='black')
plt.scatter(clusters[3][0], clusters[3][1], marker='*', s=200, color='black')

# Show final graph
plt.show()
