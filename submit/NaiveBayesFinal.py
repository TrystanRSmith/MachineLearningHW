import numpy as np
import pandas as pd
import re
import os
import random

os.chdir("C:/Users/Trystan/Documents/MachineLearning/HW4")
# for i in range(20):
# def getTraining(fileNumber):

languages = ("e", "j", "s")
characters = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " ")

N = 30 # 3 * 10
K_L = len(languages)
K_S = len(characters)
a = 0.5 #alpha
priorList = np.zeros(len(languages))
conditionalList = np.zeros(len(languages))



def Prior():
    for i, _ in enumerate(languages):
        counter = 0
        for j in range(10):
            counter += 1
        priorList[i] = (counter + a)/(N+K_L*a)
    return priorList

print(Prior())

def PrepareData(fileName):
    fileName = pd.read_csv(f"text/{fileName}.txt", sep=" |\n|\r|\t", index_col = False, header=None) #load data
    fileName = fileName.to_numpy() #convert to numpy array
    fileName = fileName.flatten() #flatten into vector
    fileName = fileName[fileName != np.array(None)] #mask to delete None
    for i in range(len(fileName)): # convert nan to space
        if type(fileName[i]) != str:
            fileName[i] = ' '
    # fileName = fileName[fileName != 'None'] #delete 'None' / nan 
    charList = []  
    for i in range(len(fileName)):
        string = fileName[i]
        charList.append(list(string))
    chars = sum(charList,[])
    return chars

def PrepareInput(fileStart, fileEnd, lang):
    bagData = []
    for l in lang:
        dataList = []
        for i in range(fileStart, fileEnd):
            file = l + f"{i}"
            data = PrepareData(file)
            dataList.append(data)

        wordBag = []
        for sublist in dataList:
            for item in sublist:
                wordBag.append(item)
        
        bagData.append(wordBag)
    return bagData

trainList = PrepareInput(0, 9, languages)
testList = PrepareInput(10, 19, languages)


def GetWordVectors(bagData):
    empty = []
    wordVectors =[]
    if len(bagData) == 3: #if data is a list of 3 language vectors
        for i, _ in enumerate(languages):
            charCount = []
            for c in characters:
                counter = 0
                for h in bagData[i]:
                    if c == h:
                        counter += 1
                charCount.append(counter)
            wordVectors.append(charCount)
    else:
        charCount = []
        for c in characters:
            counter = 0
            for h in bagData:
                if c == h:
                    counter += 1
            charCount.append(counter)
        wordVectors.append(charCount)
        wordVectors = sum(wordVectors, empty)
    return wordVectors

training = GetWordVectors(trainList)

def GetConditionals(listOfData):
    empty = []
    conditionals = []
    if len(listOfData) == 3: #if data is a list of 3 language vectors
        for i, _ in enumerate(languages):
            conditionalList = []
            for j in listOfData[i]:
                conditional = (j + 0.5) / (sum(listOfData[i]) + 13.5)
                conditionalList.append(conditional)
            conditionals.append(conditionalList)
    else:
        conditionalList = []
        listOfData = sum(listOfData, empty)
        for j in listOfData:
            conditional = [(j+0.5)/(np.sum(listOfData)+13.5)]
            conditionals.append(conditional)
        conditionals = sum(conditionals, empty)

    return conditionals



e10 = "e10"
e10data = PrepareData(e10) #prepare e10 in correct format
e10bag = GetWordVectors(e10data) #print bag of word for e10

condTrain = GetConditionals(training)
print(condTrain)
print(e10bag)

def GetLikelihood(cond, bag):
    likelihoodList = []
    for i, _ in enumerate(languages):
        logCond = np.log(cond[i])
        p = []
        for j in range(len(bag)):
            p.append((bag[j]*logCond[j]))
        prob = np.sum(p) #np.exp(np.sum(p))
        likelihoodList.append(prob)
        
    return likelihoodList

e10likelihood = GetLikelihood(condTrain, e10bag)
trainPrior = Prior()

def Posterior(likelihood: list, prior: list):
    posterior = []
    for i in range(len(likelihood)):
        posterior.append(prior[i] * likelihood[i])
    return posterior

print("Posterior:", Posterior(e10likelihood, trainPrior))

for l in languages:
    for i in range(10,20):
        print(f"{l}{i}:", languages[np.argmax(Posterior(GetLikelihood(condTrain, GetWordVectors(PrepareData(f"{l}{i}"))), trainPrior))])

random.shuffle(e10data)
e10shuffBag = GetWordVectors(e10data)
e10shuffLikelihood = GetLikelihood(condTrain, e10shuffBag)
print("e10 shuffled:", languages[np.argmax(Posterior(e10shuffLikelihood, trainPrior))])

