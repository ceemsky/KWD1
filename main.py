from kNN import kNN as kNN
import foos


dataLabels= foos.getDataLabels('/home/ceem/PycharmProjects/kwd1/data/iris.data.learning')
testData = foos.getDataLabels('/home/ceem/PycharmProjects/kwd1/data/iris.data.test')
data=foos.getData(testData)
labels=foos.getLabels(testData)
kNN = kNN(15, dataLabels)


temp=kNN.predict(data)
print(kNN.score(testData, temp))
