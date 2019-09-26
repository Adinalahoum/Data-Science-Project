import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns 

#Load the data
dt = pd.read_csv("Iris.csv")
#Descriptive Statistics 
summ = dt.describe()
print(summ)
#Scatterplot 
plt.scatter(dt["SepalLengthCm"], dt["SepalWidthCm"], alpha=0.5)
plt.title("First scatter plot")
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.show()

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
X = dt.iloc[:,[1,3]].values
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 20, c = 'green', label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')


plt.show()

