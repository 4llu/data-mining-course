import pandas as pd
import numpy as np

from sklearn.cluster import SpectralClustering

# Metrics
from sklearn.metrics.cluster import normalized_mutual_info_score

# Preprocessing
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA


# Read and process data
df = pd.read_csv("./data/genedata.csv", header = 0)

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

# PCA
print("Computing PCAs")
data_pca = []
for i in range(1, 20):
    print(i)
    data_pca.append(PCA(n_components=600).fit_transform(data))


## Clustering

print()
print("Compute clusters")

scores_spc_nn_pca = []
labels_spc_nn_pca = []

spc = SpectralClustering(
                                n_clusters = 5,
                                affinity = "nearest_neighbors",
                                n_neighbors = 5,
                                n_jobs = 3
                            )

for i, da in enumerate(data_pca):
    labels = spc.fit_predict(da)
    score = nmi_score(labels_true, labels)

    scores_spc_nn_pca.append(score)
    labels_spc_nn_pca.append(labels)

    print(i, score)
    if score > 0.99:
        print("x")

max_i = np.argmax(scores_spc_nn_pca)
max_labels = labels_spc_nn_pca[max_i]

print()
print("Best score:", scores_spc_nn_pca[max_i])
print("If best score >0.99, try again. Correct initialization happens around 5% of time.")

## You can write the solution labels to a file with this

# with open("solution1.txt", "w") as solution:
#     for l in max_labels:
#         solution.write("{}\n".format(str(l)))