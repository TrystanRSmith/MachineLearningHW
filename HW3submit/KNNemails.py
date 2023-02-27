import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier


import os
os.chdir("C:/Users/Trystan/Documents/MachineLearning/HW3")

Emails = pd.read_csv("data/emails.csv")
Emails = Emails.drop(Emails.columns[0], axis = 1)
y = Emails.loc[:, "Prediction"]
x = Emails.loc[:, Emails.columns != "Prediction"]

classifier = KNeighborsClassifier(n_neighbors = 1).fit(x,y)
crossValScores = cross_val_score(classifier, x, y, cv=5)
print(crossValScores)
scoringMetric = ['precision_macro', 'recall_macro']
crossValPrecRecall = cross_validate(classifier, x, y, cv=5, scoring=scoringMetric)
print(crossValPrecRecall)

testAccuracy = []
k = [1, 3, 5, 7, 10]
for n_neighbors in k:
    KNNmodel = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x,y)
    crossValScoresK = cross_val_score(KNNmodel, x, y, cv=5)
    accuracyScore = np.mean(crossValScoresK)
    testAccuracy.append(accuracyScore)

print(testAccuracy)

plt.plot(k, testAccuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.show()