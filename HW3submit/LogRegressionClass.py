import numpy as np


def SigmoidFunction(x):
    return 1/(1+np.exp(-x))


class LogisticRegression():

    def __init__(self, learningRate=0.001):
        self.learningRate = learningRate
        self.w = np.zeros(3000)
        self.b = 0
        self = self
    
    def fit(self, x: np.array, y):
        numSamples, numFeatures = np.shape(x)
        self.w = np.zeros(numFeatures)
        self.b = 0

        for _ in range(1000):
            linearPrediction = np.dot(x, self.w) + self.b
            predictions = SigmoidFunction(linearPrediction)
            dw = (1/numSamples)*np.dot(x.T, (predictions-y))
            db = (1/numSamples)*np.sum(predictions-y)
            
            self.w -= self.learningRate * dw
            self.b -= self.learningRate * db
    
    
    def Predictions(self, x):
        linearPrediction = np.dot(x, self.w) + self.b
        yPrediction = SigmoidFunction(linearPrediction)
        classPrediction =[]
        for y in yPrediction:
             if y < 0.5:
                 classPrediction.append(0)
             else:
                 classPrediction.append(1)
        classPrediction = np.asarray(classPrediction)
        return classPrediction
    
    def PredictionProbabilities(self, x):
        linearPrediction = np.dot(x, self.w) + self.b
        probabilities = SigmoidFunction(linearPrediction)
        return probabilities
    