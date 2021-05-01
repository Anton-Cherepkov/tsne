from tsne.tsne import TSNE
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    N = 50
    D = 10

    cluster1_center = np.ones(D) * 5
    cluster2_center = np.ones(D) * 0
    cluster3_center = np.ones(D) * -5

    X = np.concatenate([
        np.random.rand(N, D) + cluster1_center,
        np.random.rand(N, D) + cluster2_center,
        np.random.rand(N, D) + cluster3_center
    ], axis=0)
    X_transformed = TSNE(n_iter=1000).fit_transform(X)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5)
    plt.savefig('tsne.jpg')
