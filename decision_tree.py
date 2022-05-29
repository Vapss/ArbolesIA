import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv('playing_golf.csv', sep=',', engine='python')

X = df.drop(['class'],axis=1).values   
y = df['class'].values

#Separa el corpus cargado en el DataFrame en entrenamiento y el pruebas
print ('Separando los conjuntos de datos...')
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.2, shuffle = True, random_state=1)

#~ #Training
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

#~ #Testing
y_pred = clf.predict(X_test)

#~ #Model evaluation
print (classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

text_representation = tree.export_text(clf)
print(text_representation)
