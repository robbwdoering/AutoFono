import numpy as np
import math
import sys

#TODO: Randomly shuffle inputs.
#TODO: Generalize for any architecture composed of 1D layers (variable height and width)
#TODO: Implement a testing metric for the whole set (get new examples to use as test data)
#TODO: Set up cross validation data, and use it to optimize lambda and alpha 

MAXITER = 50

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
    correctError = np.multiply(-Y, np.log(prediction))
    incorrectError = np.multiply(1 - Y, np.log(1 - prediction))
    J = correctError - incorrectError
    J = np.sum(J)
    J = (1 / m) * J 
    
    #Simply the sum of the squares of the non-bias terms
    regElement = np.sum(np.square(Theta1[1:, :])) + np.sum(np.square(Theta2[1:, :])) 
    + np.sum(np.square(Theta3[1:, :]))

    #Final line officially adds regularization
    J += (lmbd / (2 * m) * regElement)
    #### COMPUTES J  ####

    # A1: mx170, Z2: mx25, Z3: mx10, mx1

    
    #### COMPUTES GRADIENTS ####
    theta3_Gradient = np.zeros(np.shape(Theta3))
    theta2_Gradient = np.zeros(np.shape(Theta2))
    theta1_Gradient = np.zeros(np.shape(Theta1))

    #D4 = A4 - y
    D4 = (A4 - Y)[0].reshape(m, 1)
    #D3 is a mx10 vector
    # D3 = Theta3*D4 .* sigGradient(Z3) 1x10
    D3 = np.multiply(np.dot(D4, Theta3[:-1].T), sigGradient(Z3))
    #D2 is a mx25 vector 1x25
    D2 = np.multiply(np.dot(D3, Theta2[:-1].T), sigGradient(Z2))

    theta3_Gradient = np.dot(A3.T, D4) / m
    theta2_Gradient = np.dot(A2.T, D3) / m
    theta1_Gradient = np.dot(A1.T, D2) / m
    # for i in range(0, m):
        # # (10x1)
        # # D4 must be accessed this way because it's techinally a double-wrapped array, so 
        # # technically two dimensional. Fix? No reason it has to be like this. 
        # D3 = np.multiply(np.dot(Theta3[1:, :].reshape(10, 1), D4[0, i].reshape(1,1)), 
                # sigGradient(Z3[i, :]).reshape(10, 1))
        
        # # (25x1)
        # D2 = np.multiply(np.dot(Theta2[1:, :], D3), sigGradient(Z2[i, :]).reshape(25, 1))

        # theta3_Gradient += np.dot(A3[i, :].reshape(11, 1), D4[0, i].reshape(1,1))
        # theta2_Gradient += np.dot(A2[i, :].reshape(26, 1), D3.T)
        # theta1_Gradient += np.dot(A1[i, :].reshape(170, 1), D2.T)


    # theta3_Gradient = theta3_Gradient / m
    # theta2_Gradient = theta2_Gradient / m
    # theta1_Gradient = theta1_Gradient / m
    #### COMPUTES GRADIENTS ####


    return (J, theta1_Gradient, theta2_Gradient, theta3_Gradient)

def sigGradient(inputVector):
    sig = sigmoid(inputVector)
    return np.multiply(sig, 1 - sig)

def randomizeWeights(firstDim, secDim, eps):
    return np.random.rand(firstDim * secDim).reshape((firstDim, secDim)) * (2 * eps) - eps

def gradientDescent(alpha, maxIter, X, Y, lmbd):
    ''' Gradient Descent
    PARAMS
    alpha: the learning rate to be used
    maxIter: the number of iterations to go through before stopping
    X: mx169 numpy matrix of positive and negative examples
    Y: mx1   numpy matrix of Labels 
    dimensions: a tuple of tuples that contains the dimensions of every Theta matrix

    RETURNS
    
    Theta1: 170x25 numpy matrix of weights (input -> hidden layer 1)
    Theta2: 26x10  numpy matrix of weights (hidden layer 1 -> hidden layer 2)
    Theta3: 11x1   numpy matrix of weigths (hidden layer 2 -> output)
    costHistory: 1xmaxIter numpy matrix of all the costs through the iteration of the program
    '''
    m = np.size(X, 0)

    epsilon = 0.1
    Theta1 = randomizeWeights(170, 25, epsilon)
    Theta2 = randomizeWeights(26, 10, epsilon)
    Theta3 = randomizeWeights(11, 1, epsilon)
    

    costHistory = np.zeros(maxIter)
    for i in range(0, maxIter):
        (costHistory[i], T1_Grad, T2_Grad, T3_Grad) = costFunction(X, Y, Theta1, Theta2, Theta3, lmbd)
        #if (i > 25) & (J - costHistory[-1] < 1):
         #   print('Cost Function Leveled out at iteration', i)
          #  break

        Theta1 -= (alpha * T1_Grad)
        Theta2 -= (alpha * T2_Grad)
        Theta3 -= (alpha * T3_Grad)
        print('Cost at iteration', i, ':', costHistory[i])
        if (i > 5) and abs(costHistory[i] - costHistory[i-1]) < 5:
            print('Converged early.')
            break

    return (Theta1, Theta2, Theta3, costHistory)


def loadNewData(string):
    if string.lower() == 'cv':
        posVectorFileName = 'CVPosEx.npy'
        negVectorFileName = 'CVNegEx.npy'
    elif string.lower() == 'test':
        posVectorFileName = 'TestPosEx.npy'
        negVectorFileName = 'TestNegEx.npy'
    else:
        posVectorFileName = 'TrainingPosEx.npy'
        negVectorFileName = 'TrainingNegEx.npy'



    posVectors = np.load(posVectorFileName).item()
    negVectors = np.load(negVectorFileName)
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

def main(decision, maxIter, lmbd, alpha):
    #Checks before reloading data, to save lots of time on debugging runs. 
    validDecision = False

    while not validDecision:
        validDecision = True

        if decision == 'y training':
            X = np.load('X.npy')
            Y = np.load('Y.npy')
        elif decision == 'n training':
            (X, Y) = loadnewdata('training')
        elif decision == 'y cv':
            X = np.load('X_CV.npy')
            Y = np.load('Y_CV.npy')
        elif decision == 'n cv':
            (X, Y) = loadnewdata('cv')
        elif decision == 'y test':
            X = np.load('X.npy')
            Y = np.load('Y.npy')
        elif decision == 'n test':
            (X, Y) = loadnewdata('test')
        else:
            validDecision = False
            decision = input('Invalid input. Use old Data?')

    #This line executes the rest of the nueral net, training apropriate thetas. 
    print('Starting gradient descent. WARNING: Will overwrite previously saved theta values.')
    (Theta1, Theta2, Theta3, costHistory) = gradientDescent(alpha, maxIter, X, Y, lmbd)
    print('Training complete in', maxIter, 'iterations! Saving thetas.')
    np.save('SimpleThetas.npy', np.array([Theta1, Theta2, Theta3]))
    return (Theta1, Theta2, Theta3, costHistory)

if __name__ == "__main__":
    decision = input("Use old data? y/n ")

    alpha = 0.01 #DEAFAULT VALUE
    lmbd = 1 #DEFAULT VALUE

    main(decision, MAXITER, lmbd, alpha)

