import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from LogRegressionClass import LogisticRegression

import os
os.chdir("C:/Users/Trystan/Documents/MachineLearning/HW3")

Emails = pd.read_csv("data/emails.csv")
Emails = Emails.drop(Emails.columns[0], axis = 1)
y = Emails.loc[:, "Prediction"]
x = Emails.loc[:, Emails.columns != "Prediction"]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)
y = y.to_numpy()
x = x.to_numpy()
xTrain = xTrain.to_numpy()
xTest = xTest.to_numpy()
yTrain = yTrain.to_numpy()
yTest = yTest.to_numpy()

logRegress = LogisticRegression(learningRate=0.001)
logRegress.fit(xTrain, yTrain)
yPrediction = logRegress.Predictions(xTest)


def Accuracy(prediction, test):
    return np.sum(prediction==test)/len(test)

accuracyScore = Accuracy(yPrediction, yTest)
print(accuracyScore)


# def PrecisionRecall(prediction, test): #TP/(TP+FP)
#     truePos = 0
#     falsePos = 0
#     trueNeg = 0
#     falseNeg = 0
#     for i in range(len(prediction)):
#         if prediction[i] == test[i]:
#             if prediction[i] == 1 and test[i] == 1:
#                 truePos +=1
#             else:
#                 trueNeg +=1
#         else:
#             if prediction[i] == 0 and test[i] == 1:
#                 falseNeg +=1
#             else:
#                 falsePos +=1
#     precision = (truePos / (truePos + falsePos))
#     recall = (truePos / (truePos + falseNeg))
#     return precision, recall



numFolds = 5
xFolds = np.array_split(x, numFolds)
yFolds = np.array_split(y, numFolds)

accuracyList = []
precisionList = []
recallList =[]

for i in range(numFolds):
    
    xTrainFoldList = []
    for j in range(i):
        for k in range(len(xFolds[j])):
            xTrainFoldList.append(xFolds[j][k])

    for j in range(i+1, numFolds):
        for k in range(len(xFolds[j])):
            xTrainFoldList.append(xFolds[j][k])

    xTrainFold = np.concatenate(xTrainFoldList, axis=0)
    xTrainFold = xTrainFold.reshape(4000,3000)

    yTrainFoldList = []
    for j in range(i):
        yTrainFoldList.append(yFolds[j])
    for j in range(i+1, numFolds):
        yTrainFoldList.append(yFolds[j])
    yTrainFold = np.concatenate(yTrainFoldList, axis=0)

    xTestFold = xFolds[i]
    yTestFold = yFolds[i]
    logRegress.fit(x=xTrainFold, y=yTrainFold)
    yPredictionFold = logRegress.Predictions(x=xTestFold)

    classifier = logRegress.fit(xTrainFold, yTrainFold)
    accuracy = Accuracy(yPredictionFold, yTestFold)
    precisionScore = precision_score(yTestFold, yPredictionFold)
    recallScore = recall_score(yTestFold, yPredictionFold)
    accuracyList.append(accuracy)
    precisionList.append(precisionScore)
    recallList.append(recallScore)

print(accuracyList)
print(precisionList)
print(recallList)