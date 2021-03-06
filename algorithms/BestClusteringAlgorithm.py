import numpy as np
import pandas as pd

class BestClustering:
    def __init__(self, predictedClusterings):
        self.predictedClusterings = predictedClusterings
        self.nbrOfRows = len(self.predictedClusterings)
        self.nbrOfClusterings = len(self.predictedClusterings[0])
        self.interClustDist = np.zeros((self.nbrOfClusterings, self.nbrOfClusterings), np.int)
        self.ttlNbrOfDisagreements = []
        self.bestClustering = -1

    def run(self):
        for i, firstRow in enumerate(self.predictedClusterings):
            for j in range(i+1, self.nbrOfRows):
                secondRow = self.predictedClusterings[j]
                self.updateDistMatrix(firstRow, secondRow)
        self.ttlNbrOfDisagreements = [sum(self.interClustDist[i,:]) for i in range(self.nbrOfClusterings)]
        self.bestClustering = self.predictedClusterings[:,np.argmin(self.ttlNbrOfDisagreements)]
        return self.bestClustering

    def updateDistMatrix(self, firstRow, secondRow):
        for i in range(self.nbrOfClusterings):
            valForColiEq = (firstRow[i] == secondRow[i])
            for j in range(i+1, self.nbrOfClusterings):
                if valForColiEq != (firstRow[j] == secondRow[j]):
                    self.interClustDist[i,j] += 1
                    self.interClustDist[j,i] += 1

if __name__ == '__main__':
    # With this example bellow the third column is the one with the less disagreements
    data = pd.read_csv("../data/testClusters.csv", delimiter=",", header=None)
    print(data)
    bc = BestClustering(data.values)
    print(bc.findBestCluster())
