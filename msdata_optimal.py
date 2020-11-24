import pandas as pd
import numpy as np

# Clustering methods
from sklearn.cluster import SpectralClustering

# Metrics
from sklearn.metrics.cluster import normalized_mutual_info_score

# Preprocessing
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import euclidean_distances


## Read and process data

df = pd.read_csv("./data/msdata.csv", header = 0)

labels_true = df["class"].to_numpy()
data = df.iloc[:, 2:].to_numpy()
data = data.astype(np.float)


## Helpers

# Scoring helper to ensure geometric averaging method is always used
def nmi_score(labels_true, labels_predicted):
    return normalized_mutual_info_score(labels_true, labels_predicted, average_method="geometric")


## Preprocessing

# Max abs scaling: scale each future by its max abs value. No centering
mabs_scaler = MaxAbsScaler()
data_mabs = mabs_scaler.fit_transform(data)

# Euclidian similarity
data_es_mabs = 1 / (1 + euclidean_distances(data_mabs, data_mabs))


## Clustering

# Euclidean distance similarity with mabs
scores_es_mabs = []
labels_es_mabs = []

spc = SpectralClustering(
            n_clusters = 5,
            affinity = "precomputed",
            n_jobs = 3
        )

labels = spc.fit_predict(data_es_mabs)
score = nmi_score(labels_true, labels)

print(score)

## You can write the solution labels to a file with this

# with open("solution2.txt", "w") as solution:
#     for l in labels:
#         solution.write("{}\n".format(str(l)))
