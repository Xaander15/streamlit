# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Page configuration
st.set_page_config(
    page_title="Drugs Segmentation",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('Drug.csv')

dataset = load_data()

# Sidebar Configuration
st.sidebar.title('🏪 Drugs Segmentation')
st.sidebar.markdown('Gunakan sidebar untuk memilih parameter KMeans')

# Menampilkan 10 baris teratas dari dataset
st.subheader("Sample Data")
st.dataframe(dataset.head(10))

# Statistik dasar dataset
st.subheader("Dataset Statistics")
st.write(dataset.describe())

# Cek Missing Values
st.subheader("Missing Values")
st.write(dataset.isnull().sum())

# Identifikasi dan hitung outlier menggunakan metode IQR
q1 = dataset.select_dtypes(exclude=['object']).quantile(0.25)
q3 = dataset.select_dtypes(exclude=['object']).quantile(0.75)
iqr = q3 - q1

batas_bawah = q1 - 1.5 * iqr
batas_atas = q3 + 1.5 * iqr

st.subheader("Batas Atas dan Bawah untuk Outlier")
st.write(f"Batas Atas: \n{batas_atas}")
st.write(f"Batas Bawah: \n{batas_bawah}")

# Filter outliers
outlier_filter = (dataset.select_dtypes(exclude=['object']) < batas_bawah) | (dataset.select_dtypes(exclude=['object']) > batas_atas)
outliers = dataset[outlier_filter.any(axis=1)]
st.subheader("Outliers Detected")
st.write(outliers)

# Sidebar untuk memilih jumlah cluster
n_clusters = st.sidebar.slider("Pilih jumlah cluster", min_value=2, max_value=10, value=5)

# Feature Selection (Assuming features are in the 4th and 5th columns)
# Menyaring hanya kolom numerik
numerical_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
if len(numerical_cols) < 2:
    st.error("Dataset harus memiliki setidaknya dua fitur numerik untuk clustering.")
else:
    X = dataset[numerical_cols].values

    # Menjalankan KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    y_kmeans = kmeans.fit_predict(X)

    # Menampilkan hasil clustering dalam tabel
    st.subheader("Clustered Data")
    clustered_data = dataset.copy()
    clustered_data['Cluster'] = y_kmeans
    st.dataframe(clustered_data.head())

    # Menampilkan hasil visualisasi elbow method untuk menentukan jumlah cluster
    wcss = []
    for i in range(1, 11):
        kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans_temp.fit(X)
        wcss.append(kmeans_temp.inertia_)

    st.subheader("Elbow Method")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('The Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    st.pyplot(plt)

    # Visualisasi Outlier - Boxplot
    st.subheader("Outlier Detection - Boxplot")
    for column in dataset.select_dtypes(exclude=['object']):
        st.markdown(f'**{column}**')
        plt.figure(figsize=(10, 1.5))
        sns.boxplot(data=dataset, x=column)
        st.pyplot(plt)

    # Visualisasi Cluster dalam 2D Scatter plot
    st.subheader(f"Visualisasi Clusters dengan {n_clusters} Clusters")
    plt.figure(figsize=(10, 6))

    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'gray']
    for i in range(n_clusters):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')
    plt.title(f'Clusters of Customers (n_clusters={n_clusters})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    st.pyplot(plt)

# Footer
st.sidebar.markdown("### About")
st.sidebar.info('''Aplikasi ini menggunakan KMeans untuk melakukan segmentasi pelanggan berdasarkan fitur numerik.
Dataset yang digunakan di sini adalah "Drug.csv".''')
