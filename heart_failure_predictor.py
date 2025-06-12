import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

hdf = pd.read_csv('heart.csv')

hdf.dropna()

print('Tabela:')
print(hdf)

#print('descrição:')
#print(hdf.describe())

#print('valores nulos: ')
#print(hdf.isnull().sum())

print('Possui problema Cardiaco:',(hdf['HeartDisease'] == 1).sum())
print('Não possui problema Cardiaco:',(hdf['HeartDisease'] == 0).sum())

#Gráfico simples dos portadores de doença
#sns.countplot(x = 'HeartDisease', data=hdf)
#plt.title('HeartDesease Dist')
#plt.show()

#treinamento de predição - introdução à ML

le = LabelEncoder() #transformação de dados - str em int - M/F em 0 ou 1
for col in hdf.select_dtypes(include='object').columns:
    hdf[col] = le.fit_transform(hdf[col]) #função labelEncoder para todas as colunas com dados não numéricos

#features x , y
x = hdf.drop('HeartDisease', axis=1) #x -> features
y = hdf['HeartDisease']  #y -> target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=31)

model = RandomForestClassifier(random_state= 31) #modelo de treinamento - randomForest - árvore de decisão
model.fit(x_train, y_train) #utiliza os dados de treino para aprender os padrões

y_pred = model.predict(x_test) #dados testes para predição

print('Acurácia do modelo:', accuracy_score(y_test, y_pred))  # Porcentagem de acerto
print('Dados do Classificador: \n', classification_report(y_test, y_pred))  # Métricas como precisão e recall

print('Esse é um código simples para aprendizado de funções básicas de manipulação de dados + introdução à biblioteca Scikit-Learn')
print('Ricardo Kenzo Koga - RA:187635 - FT - UNICAMP LIMEIRA')

#grafo da matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Doente', 'Doente'], yticklabels=['Não Doente', 'Doente'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show() # Para visualizar