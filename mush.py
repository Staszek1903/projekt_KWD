import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree, preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree.export import export_text

from pprint import pprint

names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gil-size', 'gil-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
data = pd.read_csv('./agaricus-lepiota.data', names=names)

#pozbywamy się wartości których nie ma
data = data[data['stalk-root'] != '?'] 

#
X = data.loc[:, data.columns != 'class']
y = data['class'].to_frame()

#wartości 0 i 1
X_enc = pd.get_dummies(X)
scaler = preprocessing.StandardScaler()
#wartości ustandaryzowane
X_std = scaler.fit_transform(X_enc)

le = preprocessing.LabelEncoder()
y_enc = le.fit_transform(y.values.ravel())

#podział na zbiór uczący i testujący
X_train, X_test, y_train, y_test = train_test_split(
    X_std,
    y_enc,
    test_size=0.99,
    stratify=y_enc,
    random_state=1001    
)
weights={0:5, 1:1}

#wagi jak drzewo ma brać pod uwagę rozkład danych wyjściowych
clf = tree.DecisionTreeClassifier(class_weight=weights)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Confusion Matrix: ", confusion_matrix(y_test, y_pred)) 
      
print ("Accuracy : ", accuracy_score(y_test,y_pred)*100) 
    
print("Report : ", classification_report(y_test, y_pred)) 

r = export_text(clf, feature_names=list(X_enc.columns))
print(r)

# plt.figure(dpi=250, figsize=[5.4, 3.8])
# tree.plot_tree(clf)
# plt.show()
