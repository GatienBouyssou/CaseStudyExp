import numpy as np
import pandas as pd

from utils import sortRowBasedOnSumPWD, buildMatrixPairWiseDist

class BallsAlgorithm:
    def __init__(self, predictedClusterings):
        self.nbrOfRows = len(predictedClusterings)
        self.bestClustering = -1
        self.matrixPairWiseDist = buildMatrixPairWiseDist(self.nbrOfRows, predictedClusterings)
        self.rowIdsSorted = sortRowBasedOnSumPWD(self.matrixPairWiseDist)

    def run(self, alpha):
        copyRowIdsSorted = self.rowIdsSorted.copy()
        finalClustering = np.zeros(self.nbrOfRows, dtype=np.int)
        clusterId = 1
        while len(copyRowIdsSorted) > 0:
            potentialCenter = copyRowIdsSorted[0]
            listOfNodeInCluster = [potentialCenter]
            listIndexToRemove = [0]
            sumDist = 0
            for i, nodeId in enumerate(copyRowIdsSorted):
                pairWiseDist = self.matrixPairWiseDist[potentialCenter, nodeId]
                if pairWiseDist <= 0.5:
                    listOfNodeInCluster.append(nodeId)
                    listIndexToRemove.append(i)
                    sumDist += pairWiseDist
            if sumDist/(len(listOfNodeInCluster)-1) <= alpha:
                listOfNodeInCluster.append(potentialCenter)
                finalClustering.put(listOfNodeInCluster, clusterId)
                copyRowIdsSorted = np.delete(copyRowIdsSorted, listIndexToRemove)
            else:
                finalClustering[potentialCenter] = clusterId # singleton cluster
            clusterId += 1
        self.bestClustering = finalClustering
        return finalClustering

if __name__ == '__main__':
    # With this example bellow the third column is the one with the less disagreements
    data = pd.read_csv("../data/testClusters.csv", delimiter=",", header=None)
    print(data)
    bc = BallsAlgorithm(data.values)
    print(bc.run(2/5))
    print(bc.bestClustering)