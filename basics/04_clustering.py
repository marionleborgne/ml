import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


# load the data
df = pd.read_csv('data/iris.csv')

# encode the flower species to numbers
le = LabelEncoder()
target_n = le.fit_transform(df.target)


# initialize feature matrix
X = df[['sepal_length_cm',
        'sepal_width_cm',
        'petal_length_cm',
        'petal_width_cm']]


# look for 3 clusters
km = KMeans(3)
km.fit(X)
centers = km.cluster_centers_
print centers

# plot the data and the clusters using 2 features
plt.figure(figsize=(10, 6))


# plot 1: color with real labels
plt.subplot(121)
plt.scatter(df.sepal_length_cm, df.petal_length_cm, c=target_n)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal length (cm)')
plt.title('True Labels')

# plot 2: color with found clusters (colors may differ)
plt.subplot(122)
plt.scatter(df.sepal_length_cm, df.petal_length_cm, c=km.labels_)
plt.scatter(centers[:, 0], centers[:, 2], marker='o', c='r', s=100)  # centroid
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal length (cm)')
plt.title('K-Means Clusters')
plt.draw()
plt.show()


# ## Exercises
# 
# 1) Discuss with your pair
# change the number of clusters using the n_clusters parameter. What happens?
# change the initialization parameters of KMeans to 'random'. What happens?
# run the clustering multiple times, do the centroid positions change?
#
# 2) Check the code in the advanced folder:
#    04_clustering.ipynb
