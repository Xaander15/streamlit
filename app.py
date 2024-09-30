# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load Dataset
df = pd.read_csv("Drug.csv")

# Check the dimension of the dataset
print(df.shape)

# Display the first and last 5 rows of the dataset
print(df.head(5))
print(df.tail(5))

# Check the data types of the columns
print(df.dtypes)

# Basic statistics of the dataset
print(df.describe())

# Identify and calculate outliers using IQR method
q1 = df.select_dtypes(exclude=['object']).quantile(0.25)
q3 = df.select_dtypes(exclude=['object']).quantile(0.75)
iqr = q3 - q1

batas_bawah = q1 - 1.5 * iqr
batas_atas = q3 + 1.5 * iqr

print(f"Batas Atas: \n{batas_atas}")
print(f"Batas Bawah: \n{batas_bawah}")

# Filter outliers
outlier_filter = (df.select_dtypes(exclude=['object']) < batas_bawah) | (df.select_dtypes(exclude=['object']) > batas_atas)
outliers = df[outlier_filter.any(axis=1)]
print("Outliers:\n", outliers)

# Check missing values
print(df.info())
print(df.isnull().sum())

# Feature Selection (Assuming features are in the 4th and 5th columns)
X = df.iloc[:, [3, 4]].values

# K-means model: Assuming a maximum of 10 possible clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Visualizing the Elbow Method to find optimal clusters
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means clustering with optimal clusters (let's assume 5 based on the elbow method)
kmeansmodel = KMeans(n_clusters=5, init='k-means++', random_state=0)
y_kmeans = kmeansmodel.fit_predict(X)

# Visualizing the Clusters
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, label=f'Cluster {i + 1}')
plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Visualizing outliers using a boxplot for numeric features
df_outlier = df.select_dtypes(exclude=['object'])
for column in df_outlier:
    plt.figure(figsize=(10, 2))
    sns.boxplot(data=df_outlier, x=column)
    plt.title(f'Boxplot of {column}')
    plt.show()
