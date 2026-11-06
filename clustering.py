from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

def run_kmeans(data, k):
    """
    Выполняет кластеризацию алгоритмом k-Means.
    """
    kmeans = KMeans(n_clusters=k, random_state=67, n_init='auto')
    kmeans.fit(data)
    
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_

def run_kmedoids(data, k):
    """
    Выполняет кластеризацию алгоритмом k-Medoids.
    """
    kmedoids = KMedoids(n_clusters=k, random_state=67, init='k-medoids++')
    kmedoids.fit(data)
    
    return kmedoids.labels_, kmedoids.cluster_centers_, kmedoids.inertia_