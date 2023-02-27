import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from mglearn.plots import plot_2d_separator
from mglearn import discrete_scatter


import os
os.chdir("C:/Users/Trystan/Documents/MachineLearning/HW3")

D2z = pd.read_csv("data/D2z.txt", sep=" ", header=None)
D2z.columns = ["X1",'X2','Y'] # this will be used to train the dataset
D2zTestDF = D2z[["X1",'X2']]
D2zTest = D2zTestDF.to_numpy()
D2ZY = D2z[['Y']]
D2ZY = np.ravel(D2ZY)

D2zTestDF = D2zTestDF.to_numpy()

def oneNN(dataFrame,v):
    dataFrame["Distance"] = np.sqrt((dataFrame["X1"]-v[0])**2 + (dataFrame["X2"]-v[1])**2)
    return dataFrame.iloc[[dataFrame.idxmin()["Distance"]]]

points = []
for i in np.arange(-2,2.1,0.1):
    for j in np.arange(-2,2.1,0.1):
        points.append([i,j,oneNN(D2z,[i,j]).iloc[0]["Y"]])

        
nearestNeighborDF = pd.DataFrame(points)
nearestNeighborDF.columns = ["X1","X2","y"]

fig, ax = plt.subplots()

classifier = KNeighborsClassifier(n_neighbors = 1).fit(D2zTest,D2ZY)
plot_2d_separator(classifier, D2zTest, fill=True, ax=ax)
discrete_scatter(D2zTest[:,0], D2zTest[:,1], D2ZY, ax=ax)
D2z.plot.scatter(x="X1",y="X2",c="y",colormap="RdBu", ax=ax)
nearestNeighborDF.plot.scatter(x="X1",y="X2",c="y",colormap="RdBu", alpha=0.1, ax=ax)
ax.set_xticks(np.arange(-2,2,0.1))
ax.set_yticks(np.arange(-2,2,0.1))
plt.show()
