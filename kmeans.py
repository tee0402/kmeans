import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Generate data points
N = 60
points = np.array([[random.gauss(0, 0.6), random.gauss(0, 0.6)] for i in range(N)])
labels = []

for i in range(int(N / 3)):
    points[i][0] += 3
    points[i][1] += 3
    labels.append(0)

for i in range(int(N / 3), 2 * int(N / 3)):
    points[i][0] += 6
    points[i][1] += 6
    labels.append(1)

for i in range(2 * int(N / 3), N):
    points[i][0] += 9
    points[i][1] += 9
    labels.append(2)

plt.scatter(points[:, 0], points[:, 1], c=labels)
plt.show()

# K-means
K = 3
cluster_centers = np.array([[random.gauss(6, 1), random.gauss(6, 1)] for i in range(K)])

plt.scatter(points[:, 0], points[:, 1], c=labels)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=['red' for i in cluster_centers])
plt.show()

cluster_changed = True
iterations = 0
labels_predicted = []

while cluster_changed:
    closest = [[] for i in range(K)]
    labels_predicted.clear()

    for i in range(N):
        d = []
        for j in range(K):
            d.append(math.sqrt((cluster_centers[j][0] - points[i][0]) ** 2 + (cluster_centers[j][1] - points[i][1]) ** 2))
        labels_predicted.append(d.index(min(d)))
        closest[d.index(min(d))].append(points[i])

    closest = [np.array(array) for array in closest]

    for j in range(K):
        # if closest[j] is empty, i.e. no points were closest to cluster center j
        if len(closest[j]) == 0:
            continue

        if cluster_centers[j][0] == np.mean(closest[j][:, 0]) and cluster_centers[j][1] == np.mean(closest[j][:, 1]):
            cluster_changed = False
        else:
            cluster_centers[j][0] = np.mean(closest[j][:, 0])
            cluster_centers[j][1] = np.mean(closest[j][:, 1])

    plt.scatter(points[:, 0], points[:, 1], c=labels)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=['red' for i in cluster_centers])
    plt.show()

    iterations += 1

print('Labels predicted:', labels_predicted)
print('Iterations:', iterations)