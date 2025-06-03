import numpy as np
import pandas as pd


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def Normalization(data):
    data_norm = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        col = data[:, i]
        data_norm[:, i] = (col - np.min(col)) / (np.max(col) - np.min(col))
    return data_norm


def mode_selector():
    print("1. kmeans_uniform\n" "2. kmeans_random\n" "3. kmeans++\n")
    while True:
        mode = input("Please select a mode:")
        if mode == "1":
            return "kmeans_uniform"
        elif mode == "2":
            return "2. kmeans_random"
        elif mode == "3":
            return "kmeans++"
        else:
            print("Illegal input! Please try again.")


def initialize_centroids(data, k, method):
    if method == "kmeans_uniform":
        max_vals = np.max(data, axis=0)
        min_vals = np.min(data, axis=0)
        centroids = np.random.uniform(
            low=min_vals, high=max_vals, size=(k, data.shape[1])
        )
        return centroids
    if method == "kmeans++":
        pass
    if method == "kmeans_random":
        indices = np.random.choice(data.shape[0], size=k, replace=False)
        return data[indices]


def assign_clusters(data, centroids):
    labels = np.zeros(data.shape[0], dtype=int)
    for i in range(data.shape[0]):
        print("c1 dist: ", euclidean_distance(data[i], centroids[0]))
        print("c2 dist: ", euclidean_distance(data[i], centroids[1]))
        if euclidean_distance(data[i], centroids[0]) > euclidean_distance(
            data[i], centroids[1]
        ):
            labels[i] = 1
        else:
            labels[i] = 0
        print("point label is: ", labels[i])
    return labels


def update_centroids(data, labels, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
            print(centroids[i])
        else:
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            centroids[i] = np.random.uniform(
                low=min_vals, high=max_vals, size=data.shape[1]
            )
    return centroids


def has_converged(old_centroids, new_centroids, tol=1e-4):
    diff = np.linalg.norm(old_centroids - new_centroids)
    return diff < tol


def kmeans(data, k, init_method, max_iters=100):
    centroids = initialize_centroids(data, k, method=init_method)
    print(f"Centroids: {centroids}")
    for i in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        print(f"New Centroids: {new_centroids}")

        with open("kmeans_log.txt", "a") as f:
            f.write(f"Iteration {i+1}:\n")
            f.write(f"Labels: {labels}\n")
            f.write(f"Centroids:\n{new_centroids}\n\n")

        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels


k = 2
df = pd.read_csv("kmeans_data.csv")
data = df.to_numpy()
features = df[["Bitter", "Acidity"]].to_numpy()
features_norm = Normalization(features)
init_method = mode_selector()
centroids, labels = kmeans(features_norm, k, init_method)
print(f"centroids: {centroids}")
print(f"labels: {labels}")
