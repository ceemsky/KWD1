from kNN import kNN as kNN
import foos


lData= foos.getDataLabels('./iris.data.learning')
tData = foos.getDataLabels('./iris.data.test')
kNN = kNN(3, lData)
wanted = foos.getData(tData)
nolbs = kNN.predict(wanted)
print("Wynik: ", kNN.score(tData,nolbs))
X=kNN.score(tData,nolbs)/len(tData)*100
print("Dokladnosc: ",round(X,2))
