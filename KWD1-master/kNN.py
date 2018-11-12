import numpy
import foos
from foos import wantedLables
import operator
class kNN:
    def __init__(self, k , dataLabels):
        self.k=k
        self.dataLabels = dataLabels
    def predict(self,dataArr):
        predictedLabels = []
        for item in range(len(dataArr)):
            distArr= []
            for learningData in range(len(self.dataLabels)):
                distArr.append((self.dataLabels[learningData], foos.getDistance(self.dataLabels[learningData],dataArr[item])))
                distArr.sort(key=operator.itemgetter(1))
            ngbrs = []
            for x in range(self.k):
                ngbrs.append(distArr[x][0])
            predictedLabels.append(wantedLables(ngbrs))
        return predictedLabels
    def score(self, dataLabels, labels):
        accurate=0
        for i in range(len(dataLabels)):
            item = dataLabels[i]
            if item[4]==labels[i]:
                accurate+=1
        return accurate





