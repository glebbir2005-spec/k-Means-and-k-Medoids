import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(data, labels, centers, title, filename):
    """
    Строит точечный график кластеров.
    """
    plt.figure(figsize=(8, 6))
    
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='tab10', s=60, legend='full')
    
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=250, marker='X', edgecolors='black', label='Центры/Медоиды')
    
    plt.title(title)
    plt.xlabel('Признак 1 (Стандартизированный)')
    plt.ylabel('Признак 2 (Стандартизированный)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_inertia(k_values, inertias_kmeans, inertias_kmedoids, title, filename):
    """
    Строит линейный график зависимости ошибки кластеризации от k (Метод Локтя).
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(k_values, inertias_kmeans, marker='o', linewidth=2, label='k-Means')
    plt.plot(k_values, inertias_kmedoids, marker='s', linewidth=2, label='k-Medoids')
    
    plt.title(title)
    plt.xlabel('Число кластеров (k)')
    plt.ylabel('Ошибка (Inertia)')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()