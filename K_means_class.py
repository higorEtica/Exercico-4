import numpy as np
import Initi_cluster as ic
import Initi_distance as dist
class KMeans:
    # Construindo Kmeans
    # clusters padrões será 3
    # Tolerância padrão será 0.01
    # max_iter padrão será 100
    # run padrão será 1
    # media padrão será 
    def __init__(self, clusters = 3, tolerance = 0.01, max_iter = 100, run = 1, init_cluster = "forgy",distance_metric = "euclidean"):
        self.clusters = clusters
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.init_cluster = init_cluster
        self.distance_metric = distance_metric

        self.run = run if init_cluster == 'forgy' else 1
    
    def fit(self, data):
        row_count, col_count = data.shape

        data_values = self.__get_values(data)
        data_labels = np.zeros(row_count)

        costs = np.zeros(self.run)
        all_clusterings = []

        for i in range(self.run):
            cluster_means = self.__initialize_cluster(data_values,row_count)

            for _ in range(self.max_iter):
                previous_mens = np.copy(cluster_means)

                distances = self.__compute_distances(data_values, cluster_means,row_count)

                data_labels = self.__label_examples(distances)

                cluster_means = self.__compute_means(data_values, data_labels, col_count )

                clusters_not_changed = np.abs(cluster_means - previous_mens) < self.tolerance
                if np.all(clusters_not_changed) != False:
                    break
            
            Data_values_with_labels = np.append(data_values, data_labels[:,np.newaxis], axis=1)

            all_clusterings.append((cluster_means, Data_values_with_labels))
            costs[i] = self.__compute_cost(data_values, data_labels, cluster_means)

        best_clustering_index = costs.argmin()
        self.cost_ = costs[best_clustering_index]
        self.labels = data_labels

        return all_clusterings[best_clustering_index]

    def __initialize_cluster(self,Data,row_count):
        if self.init_cluster == "forgy":
            return ic.forgy(Data,row_count,self.clusters)
        elif self.init_cluster == 'maxmin':
            return ic.maxmin(Data,self.clusters)
        elif self.init_cluster == 'macqueen':
            return ic.macqueen(Data,self.clusters)
        else:
            raise Exception('Precisa iniciar um cluster válido {} essse não é'.format(self.init_cluster))
    
    def __compute_distances(self,Data,cluster_means,row_count):
        if self.distance_metric == 'euclidean':
            return dist.euclidean(Data,cluster_means,row_count)
        elif self.distance_metric == 'manhattan':
            return dist.manhattan(Data,cluster_means,row_count)
        elif self.distance_metric == 'cosine':
            return dist.cosine(Data,cluster_means,row_count)
        else:
            raise Exception('Precisa de uma métrica de distância válida {} não é'.format(self.distance_metric))
    
    def __label_examples(self, distance):
        return distance.argmin(axis=1)
    
    def __compute_means(self,Data,labels,col_count):
        cluster_means = np.zeros((self.clusters,col_count))
        for cluster_means_index, _ in enumerate(cluster_means):
            cluster_elements = Data[labels == cluster_means_index]
            if len(cluster_elements):
                cluster_means[cluster_means_index,:] = cluster_elements.mean(axis = 0)
        return cluster_means
    
    def __compute_cost(self, Data,labels, cluster_means):
        cost = 0
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            cluster_elements = Data[labels == cluster_mean_index]
            cost += np.linalg.norm(cluster_elements - cluster_mean, axis=1).sum()
        return cost
    
    def __get_values(self, data):
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)

        