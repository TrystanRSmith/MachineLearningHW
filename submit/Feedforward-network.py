import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import os
import numpy as np
from torch.types import Device
from keras.datasets import mnist
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
# from torchvision.utils import make_grid
from torch.optim import Adam
import torch.nn as nn
from torch import save, load
from torch.autograd import variable
import visdom



os.chdir("C:/Users/Trystan/Documents/MachineLearning/HW4")

(train, trainLabel), (test, testLabel) = mnist.load_data() #load data

train = torch.from_numpy(train) #prepare data
trainLabel = torch.from_numpy(trainLabel)
train = torch.reshape(train, (60000, 784)) / 255
trainLabel = torch.reshape(trainLabel, (60000, )) / 255
test = torch.from_numpy(test)
testLabel = torch.from_numpy(testLabel)
test = torch.reshape(test, (10000, 784)) / 255
testLabel = torch.reshape(testLabel, (10000, )) / 255


class Linear(object):

    @staticmethod 
    def Forward(x: torch, w: torch, b: torch):
        device ='cuda' if torch.cuda.is_available() else 'cpu' #set device
        
        numExamples = x.size(dim=0) #number of examples is the first column of the data
        
        numWrows = w.size(dim=0) #get number of rows of w from size
        xReshaped = torch.reshape(x,(numExamples,-1)) #reshape x to be the correct shape
        
        xReshaped = xReshaped.to(device, dtype=torch.float64) #load it to correct device
        w = w.to(device, dtype=torch.float64)
        b = b.to(device, dtype=torch.float64)

        output = torch.matmul(xReshaped,w)+b # matrix multiplcation of w times x + b
        cache = x, w, b #cache
        return output, cache
    
    @staticmethod
    def Backward(dout: torch, cache): #dout is the upstream derivative
        x, w, b = cache
        numExamples = x.size(dim=0) #number of examples is the first column of the data
        numWrows = w.size(dim=0) #get number of rows of w from size
        wT = torch.t(w) #transpose weights
        xReshaped = torch.reshape(x,(numExamples,-1)) #reshape x to be the correct shape
        xReshaped = xReshaped.to(dtype=torch.float64)
        dout = dout.to(dtype=torch.float64)
        print("dout:", dout.shape, "wT", w.T.shape, "x:", xReshaped.shape)
        print(dout.size(dim=1))
        if dout.size(dim=1) != 784: #last loop
            dx = torch.reshape(torch.matmul(dout,wT),(x.size()))
        else:
            dx = None
        if dout.size(dim=1) == 784 and xReshaped.size(dim=1) == 784: #had to change some things to get the dimensions to work out
            dw = torch.matmul(xReshaped,dout.T)
        else:
            dw = torch.matmul(xReshaped,dout)
        db = dout.sum(axis=0)
        return dx, dw, db
    
class Sigmoid(object):

    def Forward(x):
        output = 1/(1+torch.exp(-x))
        cache = x #cache the input
        return output, cache
    
    def Backward(dout, cache):
        x = cache[0]
        dx = (x) * (1-x) #s(1-s)
        #add derivative of sigmoid
        return dx
    
class ThreeLayerNet(object):

    def __init__(self, inputDimension=784, hiddenDimension=(200,100), numClass=10, wScale=0.001, dtype=torch.float32, device='cuda'): #subject to change
        self.parameters = {} #dictionary for parameters
        self.device= device
        self.parameters['W1'] = wScale * torch.randn(inputDimension, hiddenDimension[0], dtype=dtype, device=device)
        self.parameters['W2'] = wScale * torch.randn(hiddenDimension[0], hiddenDimension[1], dtype=dtype, device=device)
        self.parameters['W3'] = wScale * torch.randn(hiddenDimension[0], numClass, dtype=dtype, device=device)
        self.parameters['b1'] = torch.zeros(hiddenDimension[0], dtype=dtype, device=device)
        self.parameters['b2'] = torch.zeros(hiddenDimension[1], dtype=dtype, device=device)
        self.parameters['b3'] = torch.zeros(numClass, dtype=dtype, device=device)

    def Loss(self, x, y=None):

        y = y.to(dtype=torch.int64)
        forward1, cache1 = Linear.Forward(x, self.parameters['W1'], self.parameters['b1'])
        hidLayer1, _ = Sigmoid.Forward(forward1)
        forward2, cache2 = Linear.Forward(hidLayer1, self.parameters['W2'], self.parameters['b2'])
        hidLayer2, _ = Sigmoid.Forward(forward2)
        scores, cache3 = Linear.Forward(hidLayer2, self.parameters['W3'], self.parameters['b3'])
        print('scores:', scores.size(dim=0))
        mode = 'test' if y is None else 'train'
        if mode == 'test':
            return scores

        shifted = scores - scores.max(dim=1, keepdim=True).values
        Z = shifted.exp().sum(dim=1, keepdim=True)
        logProb = shifted - Z.log()
        probs = logProb.exp()
        numExamples = scores.size(dim=0)
        print("line 120", torch.arange(numExamples).shape, y.shape)
        loss = (-1.0 / numExamples) * logProb[torch.arange(numExamples), y].sum()
        dsoft = probs.clone()
        dsoft[torch.arange(numExamples), y] -= 1
        dsoft /= numExamples

        grads = {} #initialize the gradient dictionary
        dhidden2, grads['W3'], grads['b3'] = Linear.Backward(dsoft, cache1)
        dx2 = Sigmoid.Backward(dhidden2, cache2)
        dhidden1, grads['W2'], grads['b2'] = Linear.Backward(dx2, cache2)
        dx1 = Sigmoid.Backward(dhidden1, cache1)
        _, grads['W1'], grads['b1'] = Linear.Backward(dx1, cache3)
        
        return loss, grads

class Optimize(object):
    
    def __init__(self, model, **kwargs):
        # define parameters  
        self.model = model
        self.train = train
        self.trainLabel = trainLabel
        self.test = test
        self.testLabel = testLabel
        self.epoch = 0
        self.bestValAcc = 0
        self.bestParamaters = {}
        self.lossHistory = []
        self.trainHistory = []
        self.valHistory = []
        self.learningRate = kwargs.pop("learning rate", 0.01)
        self.batchSize = kwargs.pop("batch size", 100)
        self.epochs = kwargs.pop("epochs", 10)
        self.numSamples = kwargs.pop("number of samples", 60000)
        self.numVal = kwargs.pop("number of validation samples", 10000)
        self.device = kwargs.pop("device", "cuda")

    def Step(self):
        numTrain = self.train.shape[0]
        X = self.train[torch.randperm(numTrain)[: self.batchSize]].to(self.device)
        y = self.trainLabel[torch.randperm(numTrain)[: self.batchSize]].to(self.device)
        print(y.shape)
        print(X.shape)

        # Compute loss and gradient
        loss, grads = self.model.Loss(X, y)
        self.lossHistory.append(loss.item())

        # Perform a parameter update
        for i, j in self.model.parameters.items():
            print(i, j)
            dw = grads[i]
            self.model.parameters[i] = self.StochasticGradientDescent(j, dw)

    def StochasticGradientDescent(self, w, dw):
        print(w.shape, dw.shape)
        w = w.to(dtype=torch.float64)
        dw = dw.to(dtype=torch.float64)
        w = w-(self.learningRate * dw)
        return w
    
    def Accuracy(self, X, y, batchSize = 100):

        numExamples = X.shape[0]
        X = X[torch.randperm(numExamples, device=self.device)[:numExamples]]
        y = y[torch.randperm(numExamples, device=self.device)[:numExamples]]
        
        X = X.to(self.device)
        y = y.to(self.device)

        # Compute predictions in batches
        numBatches = numExamples // batchSize
        if numExamples % batchSize != 0:
            numBatches += 1
        predictions = []
        for i in range(numBatches):
            start = i * batchSize
            end = (i + 1) * batchSize
            scores = self.model.Loss(X[start:end])
            predictions.append(torch.argmax(scores, dim=1))

        predictions = torch.cat(predictions)
        accuracy = (predictions == y).to(torch.float).mean()

        return accuracy.item()
    
    def Training(self):

        numTrain = self.train.shape[0]
        iterations = numTrain // self.batchSize #iterations per epoch
        iterationTotal = self.epochs * iterations #total iterations
        iterations = self.epochs * numTrain
        for t in range(iterationTotal):
            self.Step()
            if t == 0 | (t == iterationTotal - 1) | (t + 1) % iterations == 0:
                trainAcc = self.Accuracy(self.train, self.trainLabel)
                self.trainHistory.append(trainAcc)
                valAcc = self.Accuracy(self.test, self.testLabel)
                self.valHistory.append(valAcc)

                if valAcc > self.bestValAcc: #replace with better model
                    self.bestValAcc = valAcc
                    self.bestParamaters = {}
                    for i, j in self.model.parameters.items():
                        self.bestParamaters[i] = j.clone()

        
        self.model.parameters = self.bestParamaters

# threeLayer = ThreeLayerNet(hiddenDimension=(200,100))
# optimizer = Optimize(model=threeLayer, epochs=10)
# optimizer.Training()

train = datasets.MNIST(root='',download=False, train=True, transform=ToTensor())
dataset = DataLoader(train, 128)

#with library
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=28*28,
                                            out_features=300)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(in_features=300,
                                            out_features=200)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(in_features=200, 
                                            out_features=10)
        self.soft = nn.Softmax()
        
    def _init_weights(self, module):
         if isinstance(module, nn.Linear):
            module.weight.data.random_(from_=-1, to=1) # change to zero
   

    def forward(self, images):
        x = images.view(-1,784)
        x = self.sigmoid1(self.linear1(x))
        x = self.sigmoid2(self.linear2(x))
        x = self.linear3(x)
        return x
    
classifier = ImageClassifier()
optim = Adam(classifier.parameters(), lr= 0.001)
crossLoss = nn.CrossEntropyLoss()


epochs = 5
iterations = 0

lossHistory =[]
for e in range(epochs):
    for i, (images,labels) in enumerate(dataset):
        out = classifier(images)
        
        classifier.zero_grad()
        loss = crossLoss(out, labels)
        loss.backward()

        optim.step()

        iterations +=1
        lossHistory.append(loss.item())

print(lossHistory)

plt.plot(lossHistory)
plt.title("learning curve w from -1 to 1")
plt.show()

