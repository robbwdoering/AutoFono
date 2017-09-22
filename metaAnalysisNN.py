import simpleNN
import numpy as np
import matplotlib.pyplot as plt




def trainNetworks():
    maxIter = 100                    #DEFAULT
    lmbd = [0, 0.2, 0.5, 0.7, 1]    #DEFAULT
    alpha = [0.1, 0.5, 1, 2]        #DEFAULT
    pairs = ()
    for alphaValue in alpha:
        for lambdaValue in lmbd:
            pairs = pairs + ((alphaValue, lambdaValue),)
    print(pairs)


    Theta1 = np.zeros((20, 170, 25))
    Theta2 = np.zeros((20, 26, 10))
    Theta3 = np.zeros((20, 11, 1))
    # Theta1 = np.load('metaAnalysisTheta1.npy')
    # Theta2 = np.load('metaAnalysisTheta2.npy')
    # Theta3 = np.load('metaAnalysisTheta3.npy')


    #costHistory = np.load('costHistory.npy')
    costHistory = np.empty((20, 100))
    print('Examples without recorded history:',np.sum(costHistory[0, :] == 0))
    finalCost = np.zeros(20)


    for count in range(0,20):
            alphaValue = pairs[count][0]
            lambdaValue = pairs[count][1] 
            print('Training Network with Alpha = {0}, Lambda = {1}'.format(alphaValue, lambdaValue))
            (Theta1[count], Theta2[count], Theta3[count], costHistory[count]) = simpleNN.main(
                    'y training', maxIter, lambdaValue, alphaValue)

            finalCost[count] = costHistory[count][-1]
            

            #This line saves all theta values as a 3 dimensional array for later access.
            np.save("metaAnalysisTheta1.npy", np.array(Theta1))
            np.save("metaAnalysisTheta2.npy", np.array(Theta2))
            np.save("metaAnalysisTheta3.npy", np.array(Theta3))
            np.save('costHistory.npy', costHistory)

            print('Saved. Iteration', str(count) + '.')




def CVTest():
    '''CV Test
    This function is built to use the various networks trained by the trainNetworks, or with some 
    minor modification any trained network with the normal interface, and test each against the CV 
    data set for comparison and evaluation purposes of various settings when training the networks.
    '''
    
    #These 3D vectors are, usually, each collections of 20 2D theta value matrices.
    Theta1 = np.load('metaAnalysisTheta1.npy')
    Theta2 = np.load('metaAnalysisTheta2.npy')
    Theta3 = np.load('metaAnalysisTheta3.npy')

    (X, Y) = simpleNN.loadNewData('CV')
    #X = np.load('XCV.npy')
    #Y = np.load('YCV.npy')
    m = np.size(X, 0)

    finalCosts = np.empty(20)
    F1 = np.empty(20)
    for i in range(0, 20):
        prediction = simpleNN.forwardProp(X, Theta1[i], Theta2[i], Theta3[i])
        print('predict[0:5]:', prediction[:5])

        correctError = np.multiply(-Y, np.log(prediction))
        incorrectError = np.multiply(1 - Y, np.log(1 - prediction))
        J = correctError - incorrectError
        J = np.sum(J)
        finalCosts[i] = J / m #the final cost, without regularization
        
        prediction = np.array(prediction > 0.5, dtype=int)
        truePos  = np.sum( np.logical_and( prediction == 1, Y == 1)) + 1
        falsePos = np.sum( np.logical_and( prediction == 1, Y == 0))
        falseNeg = np.sum( np.logical_and( prediction == 0, Y == 1))
        
        precision = np.divide(truePos, truePos + falsePos)
        recall = np.divide(truePos, truePos + falseNeg)
        
        F1[i] = (2 * precision * recall) / (precision + recall)




    print('Normal Costs,', finalCosts)
    print('F1 Scores,', F1)
    plt.plot(finalCosts)
    plt.plot(F1)
    plt.show()

    inp = input('Done?')


CVTest()
        






# userDecision = 'y'

# while userDecision == 'y':

    # userDecision = input("Look at the next graph? y/n ")
        
