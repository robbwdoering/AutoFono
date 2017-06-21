import numpy as np
import matplotlib.pyplot as pl
import math
import sys

#TODO: Randomly shuffle inputs.
#TODO: Generalize for any architecture composed of 1D layers (variable height and width)
#TODO: Implement a testing metric for the whole set (get new examples to use as test data)
#TODO: Set up cross validation data, and use it to optimize lambda and alpha 

def forwardProp(X, Theta1, Theta2, Theta3):
    ''' Cost Function
    X Param: Positive and negative Examples
    Theta1: 170x25 numpy matrix of weights (input -> hidden layer 1)
    Theta2: 26x10  numpy matrix of weights (hidden layer 1 -> hidden layer 2)
    Theta3: 11x1   numpy matrix of weigths (hidden layer 2 -> output)


    Notes:
    (4000x170) * (170x25) = (4000x25)
    (4000x26) * (26x10) = (4000x10)
    (4000x11) * (11x1) = (4000x1)
    ''' 

   
    m = np.size(X, 0)
    A1 = np.c_[np.ones(m).T, X]

    # a2 = g(z), where z = a1*theta1
    Z2 = np.dot(A1, Theta1)
    A2 = sigmoid(Z2)
    A2 = np.c_[np.ones(m).T, A2]

    # a3 = g(z), where z = a2*theta2
    Z3 = np.dot(A2, Theta2)
    A3 = sigmoid(Z3)
    A3 = np.c_[np.ones(m).T, A3]

    # a4 = g(z), where z = a3*theta3
    Z4 = np.dot(A3, Theta3)
    A4 = sigmoid(Z4)
    return A4


def sigmoid(inputVector):
    ''' Sigmoid Activation Function
    inputVector: numpy matrix of any shape
    
    Returns: a matrix of the same shape with the sigmoid function applied element wise
    '''
    divisor = np.power(math.e * np.ones(np.shape(inputVector)), -inputVector)
    return 1 / (1 + divisor)

def costFunction(X, Y, Theta1, Theta2, Theta3, lmbd):
    ''' Regularized Cost Function
    X Param: mx169 numpy matrix of positive and negative examples
    Y Param: mx1   numpy matrix of Labels 
    Theta1: 170x25 numpy matrix of weights (input -> hidden layer 1)
    Theta2: 26x10  numpy matrix of weights (hidden layer 1 -> hidden layer 2)
    Theta3: 11x1   numpy matrix of weigths (hidden layer 2 -> output)
    lmbd: lambda value for regularization. 0 means no regularization. 

    Returns: A 4-tuple, containing the scalar cost and three matrices of thetas. 
    '''
    #### FEEDFORWARD CODE ####
    m = np.size(X, 0)
    A1 = np.c_[np.ones(m).T, X]

    # a2 = g(z), where z = a1*theta1
    Z2 = np.dot(A1, Theta1)
    A2 = sigmoid(Z2)
    A2 = np.c_[np.ones(m).T, A2]

    Z3 = np.dot(A2, Theta2)
    A3 = sigmoid(Z3)
    A3 = np.c_[np.ones(m).T, A3]

    Z4 = np.dot(A3, Theta3)
    A4 = sigmoid(Z4)
    #### FEEDFORWARD CODE ####
    

    #### COMPUTES J  ####
    prediction = A4
    #J = (1/m) * SUM(-Y*log(h(x)) - (1 - Y)(log(1 - h(x))))
    J = (1 / m) * np.sum(np.multiply(-Y, np.log(prediction)) - np.multiply(1 - Y, np.log(1 - prediction)))
    
    #Simply the sum of the squares of the non-bias terms
    regElement = np.sum(np.square(Theta1[1:, :])) + np.sum(np.square(Theta2[1:, :])) 
    + np.sum(np.square(Theta3[1:, :]))

    #Final line officially adds regularization
    J += (lmbd / (2 * m) * regElement)
    #### COMPUTES J  ####


    
    #### COMPUTES GRADIENTS ####
    theta3_Gradient = np.zeros(np.shape(Theta3))
    theta2_Gradient = np.zeros(np.shape(Theta2))
    theta1_Gradient = np.zeros(np.shape(Theta1))


    D4 = (A4 - Y)
    for i in range(0, m):
        # (10x1)
        # D4 must be accessed this way because it's techinally a double-wrapped array, so 
        # technically two dimensional. Fix? No reason it has to be like this. 
        D3 = np.multiply(np.dot(Theta3[1:, :].reshape(10, 1), D4[0, i].reshape(1,1)), 
                sigGradient(Z3[i, :]).reshape(10, 1))
        
        # (25x1)
        D2 = np.multiply(np.dot(Theta2[1:, :], D3), sigGradient(Z2[i, :]).reshape(25, 1))

        theta3_Gradient += np.dot(A3[i, :].reshape(11, 1), D4[0, i].reshape(1,1))
        theta2_Gradient += np.dot(A2[i, :].reshape(26, 1), D3.T)
        theta1_Gradient += np.dot(A1[i, :].reshape(170, 1), D2.T)


    theta3_Gradient = theta3_Gradient / m
    theta2_Gradient = theta2_Gradient / m
    theta1_Gradient = theta1_Gradient / m
    #### COMPUTES GRADIENTS ####


    return (J, theta1_Gradient, theta2_Gradient, theta3_Gradient)

def sigGradient(inputVector):
    sig = sigmoid(inputVector)
    return np.multiply(sig, 1 - sig)

def randomizeWeights(firstDim, secDim, eps):
    return np.random.rand(firstDim * secDim).reshape((firstDim, secDim)) * (2 * eps) - eps

def gradientDescent(alpha, maxIter, X, Y):
    ''' Gradient Descent
    alpha:
    maxIter:
    X:
    Y:
    dimensions: a tuple of tuples that contains the dimensions of every Theta matrix. 
    '''
    m = np.size(X, 0)

    epsilon = 0.1
    Theta1 = randomizeWeights(170, 25, epsilon)
    Theta2 = randomizeWeights(26, 10, epsilon)
    Theta3 = randomizeWeights(11, 1, epsilon)
    

    costHistory = np.array([1000000000])
    for i in range(0, maxIter):
        (J, T1_Grad, T2_Grad, T3_Grad) = costFunction(X, Y, Theta1, Theta2, Theta3, 1)
        #if (i > 25) & (J - costHistory[-1] < 1):
         #   print('Cost Function Leveled out at iteration', i)
          #  break
        costHistory = np.concatenate((costHistory, [J]))
        Theta1 -= (alpha * T1_Grad)
        Theta2 -= (alpha * T2_Grad)
        Theta3 -= (alpha * T3_Grad)
        print('Cost at iteration', i, ':', J)

    return (Theta1, Theta2, Theta3)

def loadNewData():
    posVectors = np.load('PositiveExamples.npy').item()
    negVectors = np.load('NegativeExamples.npy')
    X = np.empty(169)

    #Creates an NUMEXAMPLES x 169 vector by unrolling each 13x13 example into a single row. 
    for name, vector in posVectors.items():
        X = np.vstack((X, vector.ravel()))
    X = X[1:] #Deletes the first empty row
    Y = np.ones(np.size(X, 0))
    lengthVect = np.empty(0)

    #The negValues are saved in a one dimesnional array, so we have to manually separate out every 169 values as distinct entries.
    m = len(negVectors)
    for i in range(0, m, 169):
        X = np.vstack((X, negVectors[i:i+169]))
        print('.', end = ''); sys.stdout.flush()
    Y = np.concatenate((Y, np.zeros(np.size(negVectors, 0)/169)))
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    print('Done!')
    return (X, Y)

def main():

    #Checks before reloading data, to save lots of time on debugging runs. 
    decision = input("Reload data? y/n ")
    if decision != 'y':
        X = np.load('X.npy')
        Y = np.load('Y.npy')
    else:
        (X, Y) = loadNewData()
    

    ###DEBUG CODE###
    print(np.shape(X))
    print(np.shape(Y))
    ###DEBUG CODE###
   #This line executes the rest of the nueral net, training apropriate thetas.  
    (Theta1, Theta2, Theta3) = gradientDescent(0.01, 500, X, Y)
    print('Theta1: ', Theta1)
    print('Theta2: ', Theta2)
    print('Theta3: ', Theta3)
    np.save('SimpleThetas.npy', np.array([Theta1, Theta2, Theta3]))

    







if __name__ == "__main__":
    main()

