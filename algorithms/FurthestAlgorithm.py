import numpy as np
import pandas as pd
from AgglomerativeAlgorithm import AgglomerativeAlgorithm

class FurthestAlgorithm(AgglomerativeAlgorithm):
    def __init__(self, predictedClusterings):
        super().__init__(predictedClusterings)
        self.clusterCenters = []
        self.currCost = float("inf")

    def initClusterList(self):
        return [i for i in range(self.nbrOfRows)]

    def convertToPrint(self):
        self.bestClustering = self.clusterList
        return self.bestClustering

    def updateClusterList(self):
        if len(self.clusterCenters) == 0:
            node1, node2 = self.findFurthestNodesInOneCluster(self.clusterList)
            self.clusterCenters = [node1, node2]
        else:
            self.clusterCenters.append(self.findNextClusterCenter())
        newClustersList = self.assignNodesToClusters()
        newCost = self.costOfTheSolution(newClustersList)
        if newCost < self.currCost:
            self.currCost = newCost
            self.clusterList = newClustersList
            return True
        self.clusterCenters.pop()
        return False

    def findFurthestNodesInOneCluster(self, cluster):
        furthestNodes = (-1, -1)
        biggestDst = 0
        for i, nodeId in enumerate(cluster):
            for j in range(i+1, len(cluster)):
                if self.matrixPairWiseDist[nodeId, cluster[j]] > biggestDst:
                    biggestDst = self.matrixPairWiseDist[nodeId, cluster[j]]
                    furthestNodes = (nodeId, cluster[j])
        return furthestNodes

    def assignNodesToClusters(self):
        newClustersList = []
        for i in range(self.nbrOfRows):
            closestCenter = np.argmin([self.matrixPairWiseDist[i, clusterCenterId] for clusterCenterId in self.clusterCenters])
            newClustersList.append(closestCenter)
        return newClustersList

    def findNextClusterCenter(self):
        nextClusterCenter = -1
        bestDistance = 0
        for i in range(self.nbrOfRows):
            if i in self.clusterCenters: continue
            sumDistToCenters = sum([self.matrixPairWiseDist[i, clusterCenterId] for clusterCenterId in self.clusterCenters])
            if sumDistToCenters > bestDistance:
                bestDistance = sumDistToCenters
                nextClusterCenter = i
        return nextClusterCenter

    def costOfTheSolution(self, newClustersList):
        costForCluster = 0
        for i, clusterId in enumerate(newClustersList):
            for j in range(i + 1, self.nbrOfRows):
                if clusterId == newClustersList[j]:
                    costForCluster += self.matrixPairWiseDist[i, j]
                else:
                    costForCluster += 1 - self.matrixPairWiseDist[i, j]
        return costForCluster


if __name__ == '__main__':
    # With this example bellow the third column is the one with the less disagreements
    data = pd.read_csv("../data/testClusters.csv", delimiter=",", header=None)
    print(data)
    fa = FurthestAlgorithm(data.values)
    print(fa.run())
    print(fa.bestClustering)