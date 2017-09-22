import numpy as np
import math
import sys

#TODO: Randomly shuffle inputs.
#TODO: Generalize for any architecture composed of 1D layers (variable height and width)
#TODO: Implement a testing metric for the whole set (get new examples to use as test data)
#TODO: Set up cross validation data, and use it to optimize lambda and alpha 

MAXITER = 300

def sigmoid(inputVector):
    ''' Sigmoid Activation Function
    sigmoid(x) = 1 / (1 + e ^ -x)
    inputVector: numpy matrix of any shape
    
    Returns: a matrix of the same shape with the sigmoid function applied element wise
    '''
    divisor = np.power(math.e * np.ones(np.shape(inputVector)), -inputVector)
    return 1 / (1 + divisor)

def sigGradient(inputVector):
    sig = sigmoid(inputVector)
    return np.multiply(sig, 1 - sig)

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

    Z3 = np.dot(A2, Theta2)
    A3 = sigmoid(Z3)
    A3 = np.c_[np.ones(m).T, A3]

    Z4 = np.dot(A3, Theta3)
    A4 = sigmoid(Z4)
    return A4

def f1CostFunction(X, Y, Theta1, Theta2, Theta3, lmbd):
    ''' Regularized Cost Function running off of the F1 measure instead of simple subtraction
    #DEBUG
    X Param: mx169 numpy matrix of positive and negative examples
    Y Param: mx1   numpy matrix of Labels 
    Theta1: 170x25 numpy matrix of weights (input -> hidden layer 1)
    Theta2: 26x10  numpy matrix of weights (hidden layer 1 -> hidden layer 2)
    Theta3: 11x1   numpy matrix of weigths (hidden layer 2 -> output)
    lmbd: lambda value for regularization. 0 means no regularization. 

    Returns: A 4-tuple, containing the scalar cost and three matrices of thetas. 
    '''

    #Feedforward code
    m = np.size(Y)
    
    A1 = np.c_[np.ones(m).T, X] #adds bias terms

    Z2 = np.dot(A1, Theta1)
    A2 = sigmoid(Z2)
    A2 = np.c_[np.ones(m).T, A2] #adds bias terms

    Z3 = np.dot(A2, Theta2)
    A3 = sigmoid(Z3)
    A3 = np.c_[np.ones(m).T, A3] #adds bias terms

    Z4 = np.dot(A3, Theta3)
    #A4 = sigmoid(Z4)

    
    #Code to compute the cost function F1
    prediction = np.array(A4 > 0.5, dtype=int)

    #Bias terms added to avoid divide by 0 errors - somewhat changes F1, but not effectively 
    truePos  = np.sum( np.logical_and( prediction == 1, Y == 1)) / m
    falsePos = np.sum( np.logical_and( prediction == 1, Y == 0)) / m
    falseNeg = np.sum( np.logical_and( prediction == 0, Y == 1)) / m
    #print('TP: %.1f FP: %.1f FN: %.1f' % (truePos, falsePos, falseNeg))

    precision = np.divide(truePos, truePos + falsePos)
    recall = np.divide(truePos, truePos + falseNeg)

    if precision + recall == 0:
        print("ERROR DIVIDEBY0")

    F1 = (2 * precision * recall) / (precision + recall + 1)

    #Code to compute the derivative of F1 
    #derivF1 = ( (2 * precision) / (recall + (2 * precision)) ) - \
    #        ( (2 * recall) * precision / pow(recall + (2 * precision), 2) )
    derivF1 = precision + 

    #Code to compute the gradients of the weights
    theta3_Gradient = np.zeros(np.shape(Theta3))
    theta2_Gradient = np.zeros(np.shape(Theta2))
    theta1_Gradient = np.zeros(np.shape(Theta1))


    #D4 is the error in the output layer, computed by:
    #(Partial Derivative of F1 Cost Function) .* sigGradient(Z4) 
    D4 = np.multiply(derivF1, sigGradient(Z4))
    # D3 = Theta3*D4 .* sigGradient(Z3) 1x10
    D3 = np.multiply(np.dot(D4, Theta3[1:].T), sigGradient(Z3))
    #D2 is a mx25 vector 1x25
    D2 = np.multiply(np.dot(D3, Theta2[1:].T), sigGradient(Z2))

    theta3_Gradient = np.dot(A3.T, D4)
    theta2_Gradient = np.dot(A2.T, D3)
    theta1_Gradient = np.dot(A1.T, D2)
    return (F1, theta1_Gradient, theta2_Gradient, theta3_Gradient)




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
    prediction = np.array(A4 > 0.5, dtype=int)
   
    #THIS SECTION COMMENTED OUT TO DEBUG
    #J = (1/m) * SUM(-Y*log(h(x)) - (1 - Y)(log(1 - h(x))))
    # correctError = np.multiply(-Y, np.log(prediction))
    # incorrectError = np.multiply(1 - Y, np.log(1 - prediction))
    # J = correctError - incorrectError
    # J = (1 / m) * np.sum(J)

    J = prediction - Y
    print(J) #DEBUG
    
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
    #### COMPUTES GRADIENTS ####
    print(np.size(J))
    print(J)
    return (J, theta1_Gradient, theta2_Gradient, theta3_Gradient)

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

    epsilon = 30
    Theta1 = randomizeWeights(170, 25, epsilon)
    Theta2 = randomizeWeights(26, 10, epsilon)
    Theta3 = randomizeWeights(11, 1, epsilon)
    
    #neccessary to avoid odd errors when compared to (m, 1) vectors as opposed to (m,) vectors
    Y = np.reshape(Y, (m, 1))    

    costHistory = np.zeros(maxIter)
    for i in range(0, maxIter):
        
        (costHistory[i], T1_Grad, T2_Grad, T3_Grad) = f1CostFunction(X, Y, Theta1, Theta2, Theta3, lmbd)
        #if (i > 25) & (J - costHistory[-1] < 1):
         #   print('Cost Function Leveled out at iteration', i)
          #  break

        Theta1 += (alpha * T1_Grad)
        Theta2 += (alpha * T2_Grad)
        Theta3 += (alpha * T3_Grad)
        print('Cost at iteration', i, ':', costHistory[i])
        
        #This code does not work well with F1 scores.
        #if (i > 5) and abs(costHistory[i] - costHistory[i-1]) < 5:
        #    print('Converged early.')
        #    break

    return (Theta1, Theta2, Theta3, costHistory)


def loadNewData(mode):
    '''loadNewData
    Reimports the MFCC vectors and converts them into one single numpy matrix, 
    usable by this program.
    mode: The mode to load the data as. Either CV, test, or training. '''
    if mode.lower() == 'cv':
        posVectorFileName = 'CVPosEx.npy'
        negVectorFileName = 'CVNegEx.npy'
    elif mode.lower() == 'test':
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

    #The negValues are saved in a one dimesnional array, so we have to manually separate out every 169 values as distinct entries.
    m = len(negVectors)
    for i in range(0, m, 169):
        X = np.vstack((X, negVectors[i:i+169]))
        print('.', end = ''); sys.stdout.flush()
    Y = np.concatenate((Y, np.zeros(np.size(negVectors, 0)/169)))
    np.save('X{0}.npy'.format(mode), X)
    np.save('Y{0}.npy'.format(mode), Y)
    print('Done!')
    return (X, Y)

def main(decision, maxIter, lmbd, alpha):
    #Checks before reloading data, to save lots of time on debugging runs. 
    validDecision = False

    while not validDecision:
        validDecision = True

        if decision == 'y training':
            X = np.load('XTraining.npy')
            Y = np.load('YTraining.npy')
        elif decision == 'n training':
            (X, Y) = loadNewData('Training')
        elif decision == 'y cv':
            X = np.load('XCV.npy')
            Y = np.load('YCV.npy')
        elif decision == 'n cv':
            (X, Y) = loadNewData('CV')
        elif decision == 'y test':
            X = np.load('XTest.npy')
            Y = np.load('YTest.npy')
        elif decision == 'n test':
            (X, Y) = loadNewData('Test')
        else:
            validDecision = False
            decision = input('Invalid input. Use old Data? y/n ')

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
