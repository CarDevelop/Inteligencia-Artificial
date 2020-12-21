 # -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:49:46 2020

@author: 59165
"""
#librerias
import numpy as np
import pandas as pd
#librerias para aplicar los algoritmos
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")


print(df_train.head())
print(df_test.head())
#veruificar la cantidad de datos que hay
print(df_train.shape)
print(df_test.shape)

#verificamos el tipo de datos contenido en ambos datasets
print("TIPOS DE DATOS")
print(df_train.info())
print(df_test.info())
#verifica los datos faltantes de los datasets
print("DATOS FALTANTES")
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

#verifica las estadisticas del dataset
print("ESTADISTICAS DEL DATASET")
print(df_train.describe())
print(df_test.describe())

#**********PRESPROCESAMIENTO DE LA DATA********
#cambio de datos de sexos en nuemros
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)
df_test['Sex'].replace(['female','male'],[0,1],inplace=True)
#cambio los datos de embarque en numeros
df_train['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)
df_test['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)

#reemplazo de datos faltantes en edad por la media
print(df_train["Age"].mean())
print(df_test["Age"].mean())
promedio=30
df_train["Age"]=df_train["Age"].replace(np.nan,promedio)
df_test["Age"]=df_test["Age"].replace(np.nan,promedio)


#crea varios grupos de acuerdo a bandas de las edades
bins=[0,8,15,18,25,40,60,100]
names=['1','2','3','4','5','6','7']
df_train['Age']=pd.cut(df_train['Age'],bins,labels=names)
df_test['Age']=pd.cut(df_test['Age'],bins,labels=names)

#elimina la columna de "Cabin " ya que tiene muchos datos perdidos
df_train.drop(['Cabin'],axis=1,inplace=True)
df_test.drop(['Cabin'],axis=1,inplace=True)

#elimina las columnas que se consideran que bnos son necesarias para el analisis
df_train=df_train.drop(['PassengerId','Name','Ticket'],axis=1)
df_test=df_test.drop(['Name','Ticket'],axis=1)
#se elimina las fila con los datos persidos
df_train.dropna(axis=0,how='any',inplace=True)
df_test.dropna(axis=0,how='any',inplace=True)
#verifica los datos
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

print(df_train.shape)
print(df_test.shape)

print(df_test.head())
print(df_train.head())

#***************APLICANDO ALGORITMOS DE MACHINE LEARNING********
#Separa la columna con la informacion de los sobrevivientes
X=np.array(df_train.drop(['Survived'],1))# se encuentra 
y=np.array(df_train['Survived'])


#Separa los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#Regresion logistica
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
Y_pred=logreg.predict(X_test)

print('Regresion logistica')
print(logreg.score(X_train, y_train))

#Support vector machine
svc=SVC()
svc.fit(X_train, y_train)
Y_pred=svc.predict(X_test)
print('Precision soperte de vectores')
print(svc.score(X_train, y_train))

# k vecinos
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
Y_pred=knn.predict(X_test)
print('Presicion vecinos mas cercanos')
print(knn.score(X_train,y_train))


#**********PREDICCION UZANDO MODELSO********
ids=df_test['PassengerId']
prediccion_knn=knn.predict(df_test.drop('PassengerId',axis=1))
out_knn=pd.DataFrame({'PassengerId':ids,'Survived':prediccion_knn})
print('Prediccion k vecinos')
print(out_knn.head())