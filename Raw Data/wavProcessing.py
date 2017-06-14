import scipy.io.wavfile as wav
import glob
import numpy as np

names = np.empty(0)
for fileName in glob.glob('.\wav\*'):
    (rate,sig) = wav.read(fileName)
    winLen = int(0.16 * rate)
    for index in range(winLen, len(sig), winLen):
        newName = ('.\Processed' + fileName[5:-4] + str(index/winLen) + '.wav')
        wav.write(newName, rate, sig[index - winLen:index])
    names = np.append(names, [fileName])
np.save('.\ProcessedNames.npy', names)

