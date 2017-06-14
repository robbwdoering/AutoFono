import glob
import subprocess
import os.path
 
for fileName in glob.glob('.\mp3\*'):
    print(fileName)
    wavName =  ('.\wav\\' + fileName[6:-3] + 'wav')
    if !(os.path.isfile(wavName)):
        subprocess.call(['ffmpeg', '-i', fileName, wavName])
