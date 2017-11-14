import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Generate data points
N = 60
K_actual = 3
cluster_size = int(N / K_actual)
sigma = 0.7
points = []
labels = []

for i in range(N):
    if i < cluster_size:
        points.append(np.array([random.gauss(3, sigma), random.gauss(3, sigma)]))
        labels.append(0)
    elif N / cluster_size <= i < 2 * cluster_size:
        points.append(np.array([random.gauss(6, sigma), random.gauss(6, sigma)]))
        labels.append(1)
    else:
        points.append(np.array([random.gauss(9, sigma), random.gauss(9, sigma)]))
        labels.append(2)

points = np.array(points)

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
    labels_predicted = []

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

        if cluster_centers[j][0] == np.mean(closest[j][:, 0]) and cluster_centers[j][1] == np.mean(closest[j][:, 1]) and j == 0:
            cluster_changed = False
        elif cluster_centers[j][0] != np.mean(closest[j][:, 0]) or cluster_centers[j][1] != np.mean(closest[j][:, 1]):
            cluster_changed = True
            cluster_centers[j][0] = np.mean(closest[j][:, 0])
            cluster_centers[j][1] = np.mean(closest[j][:, 1])

    plt.scatter(points[:, 0], points[:, 1], c=labels)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=['red' for i in cluster_centers])
    plt.show()

    iterations += 1

correctly_labeled_points = 0
sum_labels_predicted = [[0, 0, 0] for i in range(K)]

for i in range(K):
    for j in range(cluster_size):
        if labels_predicted[j + i * cluster_size] == 0:
            sum_labels_predicted[i][0] += 1
        elif labels_predicted[j + i * cluster_size] == 1:
            sum_labels_predicted[i][1] += 1
        elif labels_predicted[j + i * cluster_size] == 2:
            sum_labels_predicted[i][2] += 1
    max_sum = max(sum_labels_predicted[i][0], sum_labels_predicted[i][1], sum_labels_predicted[i][2])
    if sum_labels_predicted[i][0] == max_sum:
        correctly_labeled_points += sum_labels_predicted[i][0]
    elif sum_labels_predicted[i][1] == max_sum:
        correctly_labeled_points += sum_labels_predicted[i][1]
    elif sum_labels_predicted[i][2] == max_sum:
        correctly_labeled_points += sum_labels_predicted[i][2]

print('Labels predicted:', labels_predicted)
print('Correctly labeled: ', correctly_labeled_points, '/', N, ', ', round(correctly_labeled_points / N * 100, 2), '%', sep='')
print('Iterations:', iterations)