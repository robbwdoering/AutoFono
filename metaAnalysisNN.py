import simpleNN
import numpy as np
import matplotlib.pyplot as plt

maxIter = 30
lmbd = [0, 0.2, 0.5, 0.7, 1]
alpha = [0.1, 0.5, 1, 2]
pairs = ()
for alphaValue in alpha:
    for lambdaValue in lmbd:
        pairs = pairs + ((alphaValue, lambdaValue),)
print(pairs)


# Theta1 = np.zeros((20, 170, 25))
# Theta2 = np.zeros((20, 26, 10))
# Theta3 = np.zeros((20, 11, 1))
Theta1 = np.load('metaAnalysisTheta1.npy')
Theta2 = np.load('metaAnalysisTheta2.npy')
Theta3 = np.load('metaAnalysisTheta3.npy')


costHistory = np.load('costHistory.npy')
print('Examples without recorded history:',np.sum(costHistory == 0))
finalCost = np.zeros(20)


for count in [4, 19]:
        alphaValue = pairs[count][0]
        lambdaValue = pairs[count][1] 
        print('Training Network with Alpha = {0}, Lambda = {1}'.format(alphaValue, lambdaValue))
        (Theta1[count], Theta2[count], Theta3[count], costHistory[count]) = simpleNN.main('y training', maxIter, lambdaValue, alphaValue)

        finalCost[count] = costHistory[count][-1]
        

        #This line saves all theta values as a 3 dimensional array for later access.
        np.save("metaAnalysisTheta1.npy", np.array(Theta1))
        np.save("metaAnalysisTheta2.npy", np.array(Theta2))
        np.save("metaAnalysisTheta3.npy", np.array(Theta3))
        np.save('costHistory.npy', costHistory)

        print('Saved. Iteration', str(count) + '.')


plt.plot(finalCost)
plt.show()

# userDecision = 'y'

# while userDecision == 'y':

    # userDecision = input("Look at the next graph? y/n ")
    
