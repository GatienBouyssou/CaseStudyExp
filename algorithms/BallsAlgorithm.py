import numpy as np
import pandas as pd

def getPairWiseDistance(firstRow, secondRow):
    nbrOfDisagreement = 0
    for i, clusterId in enumerate(firstRow):
        if clusterId != secondRow[i]:
            nbrOfDisagreement += 1
    return nbrOfDisagreement/len(firstRow)


def sortRowBasedOnSumPWD(matrixPairWiseDist):
    return np.argsort([sum(PWDForRow) for PWDForRow in matrixPairWiseDist])


class BallsAlgorithm:
    def __init__(self, predictedClusterings):
        self.nbrOfRows = len(predictedClusterings)
        self.bestClustering = -1
        self.matrixPairWiseDist = self.buildMatrixPairWiseDist(predictedClusterings)
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

    def buildMatrixPairWiseDist(self, predictedClusterings):
        matrix = np.zeros((self.nbrOfRows, self.nbrOfRows))
        for i, firstRow in enumerate(predictedClusterings):
            for j in range(i+1, self.nbrOfRows):
                distance = getPairWiseDistance(firstRow, predictedClusterings[j])
                matrix[i,j] = distance
                matrix[j,i] = distance
        return matrix

if __name__ == '__main__':
    # With this example bellow the third column is the one with the less disagreements
    data = pd.read_csv("../data/testClusters.csv", delimiter=",", header=None)
    print(data)
    bc = BallsAlgorithm(data.values)
    print(bc.run(2/5))
    print(bc.bestClustering)