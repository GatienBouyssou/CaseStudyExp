from utils import buildMatrixPairWiseDist
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class AbstractLocalSearch(ABC):
    @abstractmethod
    def run(self):
        pass

    def findBestCluster(self, indexCurrRow, availableClustId, clusterList):
        sizeCostPerCluster = self.computeDistancesPerCluster(indexCurrRow, clusterList)
        bestClusterId = -1
        bestClusterScore = float("inf")
        for clusterId, _, _ in sizeCostPerCluster:
            finalCost = 0
            for clusterId2, size, cost in sizeCostPerCluster:
                if clusterId == clusterId2:
                    finalCost += cost
                    continue
                finalCost += size - cost
            if finalCost < bestClusterScore:
                bestClusterId = clusterId
                bestClusterScore = finalCost
        costSing = sum([size - cost for clusterId2, size, cost in sizeCostPerCluster])
        if costSing < bestClusterScore:
            bestClusterId = availableClustId
            availableClustId+=1
        return bestClusterId

    @abstractmethod
    def computeDistancesPerCluster(self, indexCurrRow, clusterList):
        pass


class LocalSearch(AbstractLocalSearch):
    def __init__(self, predictedClusterings, nbrOfClusters=5):
        self.nbrOfRows = len(predictedClusterings)
        self.matrixPairWiseDist = buildMatrixPairWiseDist(self.nbrOfRows, predictedClusterings)
        self.clusterList = [int(np.random.rand() * nbrOfClusters) for i in range(self.nbrOfRows)]
        self.availableClustId = nbrOfClusters
        self.bestClustering = -1

    def run(self):
        changeHasBeenMade = True
        while changeHasBeenMade:
            changeHasBeenMade = False
            for i, clusterId in enumerate(self.clusterList):
                bestClusterId = self.findBestCluster(i, self.availableClustId, self.clusterList)
                if bestClusterId != clusterId:
                    self.clusterList[i] = bestClusterId
                    changeHasBeenMade = True
        return self.clusterList

    def computeDistancesPerCluster(self, indexCurrRow, clusterList):
        clusterId_SizeCost = {}
        for i, clusterId in enumerate(clusterList):
            if i == indexCurrRow: continue
            if clusterId in clusterId_SizeCost:
                clusterId_SizeCost[clusterId][0] += 1 # increment the size
                clusterId_SizeCost[clusterId][1] += self.matrixPairWiseDist[indexCurrRow, i] # sum up the cost
            else:
                # associate a cluster Id to the size of the cluster and the cost to put an object in this cluster
                clusterId_SizeCost[clusterId] = [1,self.matrixPairWiseDist[indexCurrRow, i]]
        return [(clusterId, sizeCost[0], sizeCost[1]) for clusterId, sizeCost in clusterId_SizeCost.items()]

if __name__ == '__main__':
    # With this example bellow the third column is the one with the less disagreements
    data = pd.read_csv("../data/testClusters.csv", delimiter=",", header=None)
    print(data)
    ls = LocalSearch(data.values)
    print(ls.run())
