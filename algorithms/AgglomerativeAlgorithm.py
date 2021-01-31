from algorithms.utils import buildMatrixPairWiseDist
import numpy as np
import pandas as pd

class AgglomerativeAlgorithm:
    def __init__(self, predictedClusterings):
        self.nbrOfRows = len(predictedClusterings)
        self.bestClustering = -1
        self.matrixPairWiseDist = buildMatrixPairWiseDist(self.nbrOfRows, predictedClusterings)
        self.clusterList = self.initClusterList()

    def initClusterList(self):
        return [[i] for i in range(self.nbrOfRows)]

    def run(self):
        clusterHasBeenMerged = True
        while clusterHasBeenMerged:
            clusterHasBeenMerged = self.updateClusterList()
        return self.convertToPrint()

    def convertToPrint(self):
        self.bestClustering = np.ones(self.nbrOfRows, dtype=np.int)
        currClusterId = 2
        for itemInCluster in self.clusterList[1:]:
            self.bestClustering.put(itemInCluster, currClusterId)
            currClusterId += 1
        return self.bestClustering

    def avgDistBtwClusters(self, firstClusters, secondClusters):
        sumDist = 0
        for firstItemId in firstClusters:
            for secondItemId in secondClusters:
                sumDist += self.matrixPairWiseDist[firstItemId, secondItemId]
        return sumDist/(len(firstClusters) * len(secondClusters))

    def updateClusterList(self):
        for i, cluster in enumerate(self.clusterList):
            for j in range(i + 1, len(self.clusterList)):
                if self.avgDistBtwClusters(cluster, self.clusterList[j]) < 0.4:
                    clust1 = self.clusterList.pop(i)
                    clust2 = self.clusterList.pop(j-1)
                    self.clusterList.append(clust1 + clust2)
                    return True
        return False

if __name__ == '__main__':
    # With this example bellow the third column is the one with the less disagreements
    data = pd.read_csv("../data/testClusters.csv", delimiter=",", header=None)
    print(data)
    aga = AgglomerativeAlgorithm(data.values)
    print(aga.run())
    print(aga.bestClustering)
