import numpy as np
import matplotlib.pyplot as pl


inputVectors = np.load('inputVectors.npy').item()
X = np.empty(169)


for name, vector in inputVectors.items():
    X = np.vstack((X, vector.ravel()))
X = X[1:] #Deletes the first empty row





