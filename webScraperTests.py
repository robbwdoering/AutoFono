import numpy as np
from webScraper import saveObj, loadObji
import webbrowser


validURLs = loadObj('validURLs')

paywallList = {}

for url in sorted(validURLs):
    print(url)
    webbrowser.open(url)
    userDecision = input('Did you download it, skip it, or paywall-flag it?\nd/s/p ').lower()
    if userDecision == 'd':
        #Take it off the list
        finalURLs = removeKey(finalURLs, url)
        print('URL removed!')

    if userDecision == 'p':
        #Take it off the list and put it on the paywall list
        paywallList[url] = finalURLs[url]
        finalURLs = removeKey(finalURLs, url)
        print('URL removed and saved for later!')

    #if it doesn't understand, just skip that URL - the safest option
    else:
        print('URL skipped!')


saveObj('paywalledURLs', paywallList)
saveObj('finalURLs', finalURLs)





def removeKey(d, key):
    temp = dict(d)
    del temp[key]
    return temp











# Theta1 = np.load('metaAnalysisTheta1.npy')
# Theta2 = np.load('metaAnalysisTheta2.npy')
# Theta3 = np.load('metaAnalysisTheta3.npy')

# print(Theta1[:, 0, 0])
# print(Theta2[:, 0, 0])
# print(Theta3[:, 0, 0])

