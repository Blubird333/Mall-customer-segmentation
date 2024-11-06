import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customer_data = pd.read_csv('Mall_Customers.csv')

print(customer_data.isnull().sum())
print('')
print(customer_data.info())
print('')
print(customer_data.head())
print('')


X = customer_data.iloc[:,[2,4]].values

plt.scatter(X[:,0],X[:,1])
plt.xlabel('age')
plt.ylabel('spending score')

plt.show()

wcss = []

for i in range(1,11):   # to calculate wcss for each no. of clusters from 1 to 11
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1,11),wcss)
plt.title("The Elbow Point Graph")
plt.xlabel("number of clusters")
plt.ylabel("WCSS")
plt.show()


#optimum no. of clusters = 4 (see the sharp point)
# Training the K-Means clustering model

kmeans = KMeans(n_clusters=4,init="k-means++",random_state=0)

#return a label for each data point based on their clustering
Y = kmeans.fit_predict(X)
print(Y)




# plotting the 4 clusters and centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0],X[Y==0,1],s=25,c='green',label='Cluster1')
plt.scatter(X[Y==1,0],X[Y==1,1],s=25,c='blue',label='Cluster2')
plt.scatter(X[Y==2,0],X[Y==2,1],s=25,c='red',label='Cluster3')
plt.scatter(X[Y==3,0],X[Y==3,1],s=25,c='yellow',label='Cluster4')


plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=25,c='cyan',label='centroids')

plt.title('customer groupings')
plt.xlabel('age')
plt.ylabel('spending score')
plt.show()
