import numpy as np

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Função para calcular a silhueta de um ponto
def silhouette_score_point(data, labels, point_index):
    point = data[point_index]
    label = labels[point_index]
    
    # Calcular a coesão (a)
    same_cluster = data[labels == label]
    if len(same_cluster) > 1:
        a = np.mean([euclidean_distance(point, other_point) for other_point in same_cluster if not np.array_equal(point, other_point)])
    else:
        a = 0

    # Calcular a separação (b)
    other_clusters = [data[labels == other_label] for other_label in set(labels) if other_label != label]
    b = np.min([np.mean([euclidean_distance(point, other_point) for other_point in other_cluster]) for other_cluster in other_clusters])

    # Calcular a silhueta
    s = (b - a) / max(a, b)
    return s

# Função para calcular a silhueta média para o conjunto de dados
def silhouette_score(data, labels):
    scores = [silhouette_score_point(data, labels, i) for i in range(len(data))]
    return np.mean(scores)