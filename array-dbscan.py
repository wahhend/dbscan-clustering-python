import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

def findNeighbors(data, i, eps, distances):
    # return all neighbor with distance less than eps
    return np.array([d for d, distance in zip(data, distances[i]) if distance>0 and distance<eps])


def expandCluster(C, data, neighbors, eps, minPts, cluster, distances):
    # neighbors = list(neighbors)
    # for neighbor in neighbors:
    while len(neighbors) > 0:
        # take first neighbor and delete it from array
        neighbor = neighbors[0]
        neighbors = np.delete(neighbors, 0, 0)

        # if labelled as noise, assign current cluster
        if cluster[neighbor[0]] == -1:
            cluster[neighbor[0]] = C
        # if no label, assign current cluster and expand cluster to its neighbor
        elif cluster[neighbor[0]] == 0:
            cluster[neighbor[0]] = C
            nNeighbors = findNeighbors(data, neighbor[0], eps, distances)

            if len(nNeighbors) >= minPts:
                print('current neighbor', neighbor)
                print('expanded neighbor', nNeighbors)
                # add current neighbor's neighbor to original neighbor array
                neighbors = np.unique(np.concatenate((neighbors, nNeighbors), axis=0), axis=0)
            # neighbors = neighbors + nNeighbors
        print('cluster', cluster)
            

def dbscan(data, eps, minPts):
    # setup
    cluster = np.zeros(len(data))
    distances = euclidean_distances(data)
    print(distances)
    # print([i for i in range(data.shape[0])])
    data = np.insert(data, 0, [i for i in range(data.shape[0])], axis=1)
    C = 0
    # print(data)
    # iterate through all point
    for i in range(len(data)):
        print('current data', data[i])
        if cluster[i] != 0:
            continue
        
        neighbors = findNeighbors(data, i, eps, distances)
        print('neighbors', neighbors)
        if len(neighbors) < minPts:
            # if minPts is not satisfied, data -> noise
            cluster[i] = -1
        else:
            # assign new cluster to the point, and expand from that point
            C += 1
            cluster[i] = C
            expandCluster(C, data, neighbors, eps, minPts, cluster, distances)
        # print('cluster', cluster)

    return cluster


df = pd.read_csv('Sales_Transactions_Dataset_Weekly_not_Normalized_200.csv', sep=';')
df = df.drop('Product_Code', axis=1)
print(df.head())
# print(euclidean_distances(df.values))
print(dbscan(df.values, 100, 30))

# print(df.shape)
