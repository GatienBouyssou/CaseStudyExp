import numpy as np
import pandas as pd
from BallsAlgorithm import BallsAlgorithm
from LocalSearchAlgorithm import AbstractLocalSearch
from utils import getPairWiseDistance

class SamplingAlgorithm(AbstractLocalSearch):
    def __init__(self, predictedClusterings, algorithmToRun, batchSize=100, *args):
        self.nbrOfRows = len(predictedClusterings)
        self.predictedClusterings = predictedClusterings
        self.indexesChoosenForSample, self.indexesNotInSample = self.createSample(batchSize)
        if args:
            self.algorithmToRun = algorithmToRun(predictedClusterings[self.indexesChoosenForSample], args[0])
        else:
            self.algorithmToRun = algorithmToRun(predictedClusterings[self.indexesChoosenForSample])
        self.availableClustId = 10

    def createSample(self, batchSize):
        indexForSample = []
        indexNotInSample = []
        threshold = 1/batchSize
        for i in range(self.nbrOfRows):
            if np.random.rand() < threshold and len(indexForSample) <= batchSize:
                indexForSample.append(i)
            else:
                indexNotInSample.append(i)
        while len(indexForSample) < batchSize:
            for i in indexNotInSample:
                if np.random.rand() < threshold:
                    indexForSample.append(i)
                    indexNotInSample.remove(i)
                    if len(indexForSample) >= batchSize: break
        return indexForSample, indexNotInSample

    def run(self):
        try:
            bestClustering = self.algorithmToRun.run()
        except:
            raise Exception("Please give us the instance of the algorithm you want to run")
        bestClustering += self.findBestClustersForNonSamples(bestClustering)
        allIndexes = self.indexesChoosenForSample + self.indexesNotInSample
        return self.reorderIndexesOfClusterings(bestClustering, allIndexes)

    def reorderIndexesOfClusterings(self, bestClustering, allIndexes):
        finalClutering = np.zeros(self.nbrOfRows, dtype=np.int)
        for i, bestCluster in enumerate(bestClustering):
            finalClutering[allIndexes[i]] = bestCluster
        return finalClutering

    def findBestClustersForNonSamples(self, bestClusteringForSample):
        self.availableClustId = max(bestClusteringForSample)
        bestClusteringForOtherNodes = []
        for nonSampleId in self.indexesNotInSample:
            bestClustering = self.findBestCluster(nonSampleId, self.availableClustId, bestClusteringForSample)
            bestClusteringForOtherNodes.append(bestClustering)
        return bestClusteringForOtherNodes

    def computeDistancesPerCluster(self, indexCurrRow, clusterList):
        clusterId_SizeCost = {}
        for i, clusterId in enumerate(clusterList):
            if i == indexCurrRow: continue
            if clusterId in clusterId_SizeCost:
                clusterId_SizeCost[clusterId][0] += 1 # increment the size
                clusterId_SizeCost[clusterId][1] += getPairWiseDistance(self.getRowForSplId(i), self.getRowForNonSplId(indexCurrRow)) # sum up the cost
            else:
                # associate a cluster Id to the size of the cluster and the cost to put an object in this cluster
                clusterId_SizeCost[clusterId] = [1, getPairWiseDistance(self.getRowForSplId(i), self.getRowForNonSplId(indexCurrRow))]
        return [(clusterId, sizeCost[0], sizeCost[1]) for clusterId, sizeCost in clusterId_SizeCost.items()]

    def getRowForSplId(self, i):
        return self.predictedClusterings[self.indexesChoosenForSample[i]]

    def getRowForNonSplId(self, i):
        return self.predictedClusterings[i]


if __name__ == '__main__':
    # With this example bellow the third column is the one with the less disagreements
    data = pd.read_csv("../data/testClusters.csv", delimiter=",", header=None)
    print(data)
    ls = SamplingAlgorithm(data.values, BallsAlgorithm, 3, 0.4)
    print(ls.run())