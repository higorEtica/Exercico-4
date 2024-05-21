import numpy as np

def distancia_euclidiana(a, b):
    return np.linalg.norm(a - b)

def k_means(dataset, k, max_iter=100):
    # Inicialização dos centróides de forma aleatória
    centroids = dataset[np.random.choice(range(len(dataset)), size=k, replace=False)]

    for _ in range(max_iter):
        # Inicialização de clusters vazios
        clusters = [[] for _ in range(k)]

        # Atribuição de cada ponto ao cluster mais próximo
        for point in dataset:
            distances = [distancia_euclidiana(point, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        # Atualização dos centróides
        new_centroids = [np.mean(cluster, axis=0) if cluster else centroid for centroid, cluster in zip(centroids, clusters)]

        # Verificação de convergência
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # Atribuição de rótulos aos pontos
    labels = np.zeros(len(dataset), dtype=int)
    for i, cluster in enumerate(clusters):
        for point in cluster:
            index = np.where((dataset == point).all(axis=1))[0][0]
            labels[index] = i

    return np.array(centroids), labels
