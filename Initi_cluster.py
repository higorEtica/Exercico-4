import numpy as np

def forgy(Data, row_count, clusters):
    return Data[np.random.choice(row_count, clusters, replace=False)]

def macqueen(Data, clusters):
    return Data[: clusters]

def maxmin(Data, clusters):
    Data_ = np.copy(Data)
    initial_centers = np.zeros((clusters, Data_.shape[1]))
    Data_norms = np.linalg.norm(Data_, axis=1)
    Data_norms_max_1 = Data_norms.argmax()
    initial_centers[0] = Data_[np.argmax(Data_norms)]
    Data_ = np.delete(Data_,Data_norms_max_1 , axis=0)

    for i in range(1, clusters):
        distances = np.zeros((Data_.shape[0], i))
        for index, center in enumerate(initial_centers[:i]):
            distances[:, index] = np.linalg.norm(Data_ - center, axis=1)
        
        max_min_index = distances.min(axis=1).argmax()

        initial_centers[i] = Data_[max_min_index]
        Data_ = np.delete(Data_, max_min_index, axis=0)
    
    return initial_centers