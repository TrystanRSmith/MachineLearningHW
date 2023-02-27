import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from LogRegressionClass import LogisticRegression

import os
os.chdir("C:/Users/Trystan/Documents/MachineLearning/HW3")

Emails = pd.read_csv("data/emails.csv")
Emails = Emails.drop(Emails.columns[0], axis = 1)
y = Emails.loc[:, "Prediction"]
x = Emails.loc[:, Emails.columns != "Prediction"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)
xTrain = xTrain.to_numpy()
xTest = xTest.to_numpy()
yTrain = yTrain.to_numpy()
yTest = yTest.to_numpy()

logRegress = LogisticRegression(learningRate=0.001)
logRegress.fit(xTrain, yTrain)
logProb = logRegress.PredictionProbabilities(xTest)

knnClassifier = KNeighborsClassifier(n_neighbors = 5)
knnClassifier.fit(xTrain,yTrain)
knnProb = knnClassifier.predict_proba(X=xTest)[:,1]

fig, ax = plt.subplots()

RocCurveDisplay.from_predictions(yTest, knnProb, ax=ax)
RocCurveDisplay.from_predictions(yTest, logProb, ax=ax)
plt.show()


# Question1-5
Roc_question1 = ([0.33,0.66,1,1], [0,0.25,0.5,1])
plt.plot(Roc_question1[1], Roc_question1[0], '-o')
plt.xlabel('FP')
plt.ylabel('TP')
plt.title('ROC')
plt.show()