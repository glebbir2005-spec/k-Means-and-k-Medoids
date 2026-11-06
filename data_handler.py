import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath, columns=None):
    """
    Загружает датасет и проводит стандартизацию выбранных признаков.
    """
    df = pd.read_csv(filepath)
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols[-2:]
        
    data = df[columns].values
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, df, columns

def add_noise(data, noise_fraction):
    """
    Добавляет шум к данным, искажая значения для заданного процента.
    """
    if noise_fraction <= 0.0:
        return data.copy()
        
    noisy_data = data.copy()
    n_samples, n_features = noisy_data.shape
    n_noisy = int(n_samples * noise_fraction)
    
    indices = np.random.choice(n_samples, n_noisy, replace=False)
    
    noise = np.random.normal(0, 3, size=(n_noisy, n_features))
    
    noisy_data[indices] += noise
    
    return noisy_data