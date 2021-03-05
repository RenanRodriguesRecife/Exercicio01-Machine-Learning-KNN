#base de dados escolhidas
#DATATRIEVE Transition/Software defect prediction


#biblioteca para importar formato arff
from scipy.io import arff
#biblioteca para manipular os dados
import pandas as pd
#biblioteca com o knn
from sklearn.neighbors import KNeighborsClassifier
#biblioteca para fazer o kfold cross-validation
from sklearn.model_selection import cross_val_score


data = arff.loadarff('data1.arff')
df = pd.DataFrame(data[0])
#print(df.head())

X = df.drop(columns=['Faulty6_1'])
print(X)
Y = df['Faulty6_1'].values.toInt
print(Y)



#KNN normal
#knn = KNeighborsClassifier(n_neighbors=10,weights='uniform',n_jobs=-1)


#KNN Ponderado
knn = KNeighborsClassifier(n_neighbors=10,weights='distance',n_jobs=-1)


score = cross_val_score(knn, X, Y, cv=10,scoring='accuracy')

print(score)
print('chegou aqui')