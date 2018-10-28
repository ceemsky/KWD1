
import pandas
import numpy
import scipy.spatial


def getDataLabels(path):
    data = pandas.read_csv(path)
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
def getDistance(vector1,vector2):
    return scipy.spatial.distance.euclidean(vector1[0:2],vector2[0:2])
def predictLabel(labelsArr):
    labels = {"Iris-virginica":0,"Iris-versicolor":0,"Iris-setosa":0}
    for item in labelsArr:
        labels[item[1]] += 1
    sortedLabels = sorted(labels.__iter__())
    return sortedLabels[0]
