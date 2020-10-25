import math
from numpy import dot
from numpy.linalg import norm
import random
import time
from tkinter import *


def euclidean(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    dist = 0
    for i in range(1, len(instance1)):
        dist += (instance1[i] - instance2[i])**2
    return math.sqrt(dist)

def sum_of_squares(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    dist = 0
    for i in range(1, len(instance1)):
        dist += (instance1[i] - instance2[i])**2
    return dist


def cosine_sim(a, b):
    if a is None or b is None:
        return float('inf')

    return 1 - (dot(a[1:], b[1:])/(norm(a[1:])*norm(b[1:])))


def jaccard(a, b):
    if a is None or b is None:
        return float('inf')
    num = 0
    den = 0
    for i in range(1, len(a)):
        num += min(a[i], b[i])
        den += max(a[i], b[i])

    return 1 - (num / den)


def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    if (numInstances == 0):
        return
    numAttributes = len(instanceList[0])
    means = [name] + [0] * (numAttributes-1)
    for instance in instanceList:
        for i in range(1, numAttributes):
            means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    return tuple(means)


def assign(instance, centroids, metric):
    minDistance = metric(instance, centroids[0])
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = metric(instance, centroids[i])
        if (d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex


def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList


def assignAll(instances, centroids, metric):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids, metric)
        clusters[clusterIndex].append(instance)
    return clusters


def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "centroid" + str(i)
        centroid = meanInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids


def kmeans(instances, k, animation=False, initCentroids=None, metric=sum_of_squares, stop_condition='centroids'):
    if stop_condition not in ['centroids', 'SSE', 'max_iteration']:
        print('invalid stop condition, must be centroids, SSE, or max_iteration')
        return

    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        # randomly select k initial centroids
        random.seed(time.time())
        centroids = random.sample(instances, k)
    else:
        centroids = initCentroids

    if animation:
        delay = 1.0  # seconds
        canvas = prepareWindow(instances)
        clusters = createEmptyListOfLists(k)
        clusters[0] = instances
        paintClusters2D(canvas, clusters, centroids, "Initial centroids")
        time.sleep(delay)

    prevCentroids = []
    withinss = float('inf')
    prev_withinss = float('inf')
    iteration = 0
    continue_flag = True

    while continue_flag:
        iteration += 1
        clusters = assignAll(instances, centroids, metric)
        if animation:
            paintClusters2D(canvas, clusters, centroids,
                            "Assign %d" % iteration)
            time.sleep(delay)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)
        prev_withinss = withinss
        withinss = computeWithinss(clusters, centroids)
        if animation:
            paintClusters2D(canvas, clusters, centroids,
                            "Update %d, withinss %.1f" % (iteration, withinss))
            time.sleep(delay)

        # Set stop condition
        if stop_condition == 'centroids':
            continue_flag = centroids != prevCentroids
        elif stop_condition == 'SSE':
            continue_flag = withinss > prev_withinss
        else:
            continue_flag = iteration < 100

    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["iterations"] = iteration
    return result


def computeWithinss(clusters, centroids):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            result += sum_of_squares(centroid, instance)
    return result

# Repeats k-means clustering n times, and returns the clustering
# with the smallest withinss


def repeatedKMeans(instances, k, n):
    bestClustering = {}
    bestClustering["withinss"] = float("inf")
    for i in range(1, n+1):
        print("k-means trial %d," % i)
        trialClustering = kmeans(instances, k)
        print("withinss: %.1f" % trialClustering["withinss"])
        if trialClustering["withinss"] < bestClustering["withinss"]:
            bestClustering = trialClustering
            minWithinssTrial = i
    print("Trial with minimum withinss:", minWithinssTrial)
    return bestClustering