#base de dados escolhidas
#DATASET1: DATATRIEVE Transition/Software defect prediction
#DATASET2: Desharnais Software Cost Estimation

#biblioteca para importar formato arff
from scipy.io import arff
#biblioteca para manipular os dados
import pandas as pd
#biblioteca com o knn
from sklearn.neighbors import KNeighborsClassifier
#biblioteca para fazer o kfold cross-validation
from sklearn.model_selection import cross_val_score
#biblioteca para fazer gráficos
import seaborn as sns
import matplotlib.pyplot as plt
#calcular o tempo de execução
import time


'''
data = arff.loadarff('data1.arff')
df = pd.DataFrame(data[0])
#print(df.head())

X = df.drop(columns=['Faulty6_1'])
print(X)

Y = df['Faulty6_1'] = pd.to_numeric(df['Faulty6_1'],downcast='integer')
print(Y)

'''
data = arff.loadarff('data2.arff')
df = pd.DataFrame(data[0])
#remove todas as linhas que tem um atributo NAN
df = df.dropna()


X = df.drop(columns=['Language'])
print(X)

Y = df['Language'] = pd.to_numeric(df['Language'],downcast='float')

#Y = df['Language'].values
print(Y)




print('KNN')

arrayTest = []

for K in [1,2,3,5,7,9,11,13,15]:
    #KNN normal
    print ("para K = " + str(K))
    knn = KNeighborsClassifier(n_neighbors=K,weights='uniform',metric='euclidean',n_jobs=-1)
    
    start_time = time.time()
    score = cross_val_score(knn, X, Y, cv=10,scoring='accuracy')
    print("--- %s seconds ---" % (time.time() - start_time))

    print(score)
    print('média: '+ str(score.mean()))
    arrayTest.append([score.mean(),K])

graf1 = pd.DataFrame(arrayTest,columns=["acurácia média","valor de K"])


print(graf1)
sns.lineplot(data=graf1, palette="tab10", linewidth=2.5)
plt.ylim(0, 1)


plt.show()


print('KNN PONDERADO')

arrayTest = []

for K in [1,2,3,5,7,9,11,13,15]:
    
    #KNN Ponderado
    print ("para K = " + str(K))
    
    knn = KNeighborsClassifier(n_neighbors=K,weights='distance',metric='euclidean',n_jobs=-1)
    
    start_time = time.time()
    score = cross_val_score(knn, X, Y, cv=10,scoring='accuracy')
    print("--- %s seconds ---" % (time.time() - start_time))

    print(score)
    print('média: '+ str(score.mean()))
    arrayTest.append([score.mean(),K])

graf1 = pd.DataFrame(arrayTest,columns=["acurácia média","valor de K"])

print(graf1)
sns.lineplot(data=graf1, palette="tab10", linewidth=2.5)
plt.ylim(0, 1)


plt.show()
