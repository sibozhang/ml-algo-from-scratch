import numpy as np
import pandas as pd
from collections import Counter


def Normalization(data):
    data_norm = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        col = data[:, i]
        data_norm[:, i] = (col - np.min(col)) / (np.max(col) - np.min(col))
    return data_norm


def Standardization(data):
    data_std = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        col = data[:, i]
        data_std[:, i] = (col - np.mean(col)) / np.std(col)
    return data_std


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


df = pd.read_csv("knn_data.csv")
data = df.to_numpy()[:-1]
features_data = df[["Temperature", "Humidity"]].to_numpy()
labels_data = df["Mold"].to_numpy()

training_labels = labels_data[:-1]
features_data_norm = Normalization(features_data)
test_data_feature = features_data_norm[-1]

data_dist = np.zeros_like(training_labels)
for i in range(training_labels.shape[0]):
    data_dist[i] = euclidean_distance(features_data_norm[i], test_data_feature)

k = 3
nearest_indices = np.argsort(data_dist)[:k]
nearest_labels = training_labels[nearest_indices]
vote_result = Counter(nearest_labels).most_common(1)[0][0]
print(f"Predicted label for test data: {vote_result}")
print(data[nearest_indices])

print(np.__version__)
