import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances

def dbscan(data):
    data = data.values
    distances = manhattan_distances(data)
    visited = np.zeros(data.shape[0])
    indexes = [i for i in range(data.shape[0])]
    data = np.insert(data, 0, indexes, axis=1)
    idx = 2
    eps = 15
    minPts = 3
    noise = []
    print(data)
    
    while 0 in visited:
        visited[idx] = 1
        bp = np.array([d for d, dist in zip(data, distances[idx]) if dist < eps and dist > 0])
        
        if len(bp) < minPts:
            noise.append(cp)
        else:
            cp = data[idx]
            for point in bp:
                cp = point
                print(cp)
                visited[cp[0]] = 1
        print(visited)
        break

        

df = pd.read_csv('data.csv')
dbscan(df)

print(df.shape)
