import operator
import pandas
import numpy
import scipy.spatial


def getDataLabels(path):
    data = pandas.read_csv(path,header=None)
    return numpy.array(data)

def getData(dataLabelsArr):
    array= []
    for i in dataLabelsArr:
        array.append(i[0:len(i)-1])
    return numpy.array(array)
def getLabels(dataLabelsArr):
    array = []
    for i in dataLabelsArr:
        array.append(i[len(i)-1])
    return numpy.array(array)
def wantedLables(ngbrs):
    labels = {}
    for i in range(len(ngbrs)):
        label = ngbrs[i][-1]
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1
    sLabels = sorted(labels.__iter__(),key=operator.itemgetter(1))
    return sLabels[0]
def getDistance(vector1,vector2):
    return scipy.spatial.distance.euclidean(vector1[0:2],vector2[0:2])
def predictLabel(labelsArr):
    labels = {"Iris-virginica":0,"Iris-versicolor":0,"Iris-setosa":0}
    for item in labelsArr:
        labels[item[1]] += 1
    sortedLabels = sorted(labels.__iter__())
    return sortedLabels[0]

