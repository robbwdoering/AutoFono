from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import glob
import numpy as np
import matplotlib.pyplot as plt
#IDEA: Split up every incoming file into 15 frames. Vary the window size to accomplish this. 
minimum = 1000
array = np.empty(0)
for num in range(5, 50, 5):
    wavLength = np.empty(0)
    for fileName in glob.glob('.\Audio\*\*'):
        (rate,sig) = wav.read(fileName)
    #http://python-speech-features.readthedocs.io/en/latest/
        mfcc_feat = mfcc(sig, rate, num / 1000, num / 2000, 13, 26, 2048)
        wavLength = np.append(wavLength, [len(mfcc_feat)])
    array = np.append(array, [num / 1000, np.min(wavLength)])



print(array)


#plt.plot(abs(array), 'r')

