from data_handler import load_and_preprocess, add_noise
from clustering import run_kmeans, run_kmedoids
from visualizer import plot_clusters, plot_inertia

def main():
    filepath = 'customers.csv'

    print("Загрузка и предобработка данных...")
    data_scaled, df, columns = load_and_preprocess(filepath, columns=['CardDebt', 'Income'])
    print(f"Для кластеризации выбраны признаки: {columns}")

    k_values = list(range(3, 20))
    noise_levels = [0.0, 0.01, 0.03, 0.05, 0.10] 

    for noise in noise_levels:
        percent = int(noise * 100)
        if noise == 0:
            print(f"\nЭксперименты на чистых данных (0% шума)")
            current_data = data_scaled
            prefix = "clean"
        else:
            print(f"\nЭксперименты на зашумленных данных ({percent}% шума)")
            current_data = add_noise(data_scaled, noise)
            prefix = f"noise_{percent}pct"

        inertias_km = []
        inertias_kmed = []
        
        optimal_k = 5

        for k in k_values:
            labels_km, centers_km, inertia_km = run_kmeans(current_data, k)
            inertias_km.append(inertia_km)
            
            labels_kmed, centers_kmed, inertia_kmed = run_kmedoids(current_data, k)
            inertias_kmed.append(inertia_kmed)
            
            if k == optimal_k:
                plot_clusters(
                    current_data, labels_km, centers_km, 
                    f'k-Means (k={k}, Шум={percent}%)', 
                    f'{prefix}_kmeans_k{k}.png'
                )
                plot_clusters(
                    current_data, labels_kmed, centers_kmed, 
                    f'k-Medoids (k={k}, Шум={percent}%)', 
                    f'{prefix}_kmedoids_k{k}.png'
                )

        plot_inertia(
            k_values, inertias_km, inertias_kmed, 
            f'Зависимость ошибки от k (Шум: {percent}%)', 
            f'{prefix}_inertia.png'
        )
        print(f"Графики для шума {percent}% сохранены.")

    print("\nВсе эксперименты завершены.")

if __name__ == '__main__':
    main()