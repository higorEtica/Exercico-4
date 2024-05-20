import numpy as np

def euclidean(Data,cluster_means, row_count):
    distances = np.zeros((row_count, cluster_means.shape[0]))
    for index, center in enumerate(cluster_means):
        distances[:, index] = np.linalg.norm(Data - center, axis=1)
    return distances

def manhattan(Data,cluster_means, row_count):
    distances = np.zeros((row_count, cluster_means.shape[0]))
    for index, center in enumerate(cluster_means):
        distances[:, index] = np.sum(np.abs(Data - center), axis=1)
    return distances

def cosine(Data,cluster_means, row_count):
    distances = np.zeros((row_count, cluster_means.shape[0]))
    for index, center in enumerate(cluster_means):
        dot_product = np.dot(Data, center)
        data_norm = np.linalg.norm(Data, axis=1)
        center_norm = np.linalg.norm(center)
        distances[:, index] = 1 - dot_product / (data_norm * center_norm)
    return distances    