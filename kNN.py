import numpy
import foos


class kNN:
    def __init__(self, k , dataLabels):
        self.k=k
        self.dataLabels = dataLabels
    def predict(self,dataArr):
        predictedLabels = []
        for item in dataArr:
            distArr= []
            for learningData in self.dataLabels:
                distArr.append([foos.getDistance(item, learningData[0:len(learningData) - 1]), learningData[len(learningData)-1]])
                npArr=numpy.array(distArr)
                npArr.sort(0)
            predictedLabels.append(foos.predictLabel(distArr[0:self.k]))
        return predictedLabels
    def score(self, dataLabels, labels):
        accurate=0
        for i in range(len(dataLabels)):
            item = dataLabels[i]
            if item[4]==labels[i]:
                accurate+=1
        return accurate/len(dataLabels)





