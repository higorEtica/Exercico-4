import numpy as np

def euclidean_distance_matrix(data):
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = np.linalg.norm(data[i] - data[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix

def find_closest_clusters(dist_matrix):
    min_dist = np.inf
    closest_pair = (None, None)
    n = dist_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] < min_dist:
                min_dist = dist_matrix[i, j]
                closest_pair = (i, j)
    return closest_pair, min_dist

def update_distance_matrix(dist_matrix, cluster1, cluster2):
    new_dist_matrix = np.delete(dist_matrix, cluster2, axis=0)
    new_dist_matrix = np.delete(new_dist_matrix, cluster2, axis=1)
    
    for i in range(new_dist_matrix.shape[0]):
        if i != cluster1:
            new_dist_matrix[cluster1, i] = min(dist_matrix[cluster1, i], dist_matrix[cluster2, i])
            new_dist_matrix[i, cluster1] = new_dist_matrix[cluster1, i]
    
    return new_dist_matrix

def single_linkage_clustering(data, k):
    dist_matrix = euclidean_distance_matrix(data)
    clusters = [{i} for i in range(data.shape[0])]

    while len(clusters) > k:
        (cluster1, cluster2), min_dist = find_closest_clusters(dist_matrix)
        
        clusters[cluster1].update(clusters[cluster2])
        clusters.pop(cluster2)
        
       
        dist_matrix = update_distance_matrix(dist_matrix, cluster1, cluster2)

    return clusters