import numpy as np


def buildMatrixPairWiseDist(nbrOfRows, predictedClusterings):
    matrix = np.zeros((nbrOfRows, nbrOfRows))
    for i, firstRow in enumerate(predictedClusterings):
        for j in range(i + 1, nbrOfRows):
            distance = getPairWiseDistance(firstRow, predictedClusterings[j])
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def getPairWiseDistance(firstRow, secondRow):
    nbrOfDisagreement = 0
    for i, clusterId in enumerate(firstRow):
        if clusterId != secondRow[i]:
            nbrOfDisagreement += 1
    return nbrOfDisagreement/len(firstRow)


def sortRowBasedOnSumPWD(matrixPairWiseDist):
    return np.argsort([sum(PWDForRow) for PWDForRow in matrixPairWiseDist])
