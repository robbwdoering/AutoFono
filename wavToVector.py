from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import glob
import numpy as np
import matplotlib.pyplot as pl

answers = np.empty(0)
wavLength = np.empty(0)
array = np.empty(0)
for fileName in glob.glob('.\Audio\*\*'):
    (rate,sig) = wav.read(fileName)
    winLen = 0.8
    mfcc_feat = np.empty()
    while 
    winLen = (len(sig) / rate) / 1000
    mfcc_feat = mfcc(sig, rate, windowLength, windowLength / 2, 13, 26, 2048)
    array = np.append(array, [len(mfcc_feat)])
answers = np.append(answers, [np.mean(array) , windowLength])


print(answers)


#plt.plot(abs(array), 'r')
#plt.show()
