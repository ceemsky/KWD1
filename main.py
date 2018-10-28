from kNN import kNN as kNN
import foos


dataLabels= foos.getDataLabels('/home/ceem/PycharmProjects/kwd1/data/iris.data.learning')
kNN = kNN(3, dataLabels)
data= []
data.append(kNN.dataLabels[14])
data.append(kNN.dataLabels[31])
temp=kNN.predict(foos.getData(data))
print(kNN.score(data, temp))
