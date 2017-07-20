from lxml import html, etree
import requests
import numpy as np
import sys

MIN_WORD_COUNT = 100
MAX_WORD_COUNT = 2000
MIN_RR_OCCURENCE = 3
MIN_RATIO = 0.05
SEARCHED_PHONEME = 'rr'
WEBSITE_LINK = 'https://rhinospike.com'
EMAIL = 'doerinrw'
PASSWORD = 'password'

def readLangPage(num):
    '''readLangPage
    This method is built to read a single page of the "sort by language" search pages,
    with no search string given. 
    num: The index of the search page to be processed. 
    
    Returns: A dictionary, containing string URLs of valid pages,
    keyed to the ratio of trilled RRs for each.'''
    validPages = {}
    page = requests.get('https://rhinospike.com/language/spa/recordings/?page=' + str(num))
    tree = html.fromstring(page.content)
    
    
    nextTitleTreeName = '//*[@id="left_panel"]/div/div/div/table/tbody/tr/td[2]/div/span[3]/a' 
    nextWordCountTreeName = '//*[@id="left_panel"]/div/div/div/table/tbody/tr/td[2]/div[3]/span'


    for i in range(2, 12):
        #//*[@id="left_panel"]/div/div[1]/div[2]
        #//*[@id="left_panel"]/div/div[1]/div[1]

        #This block gets the title and link from the title element
        titleTree =  tree.xpath(nextTitleTreeName)
        link = WEBSITE_LINK + titleTree[0].get('href')
        title = titleTree[0].text

        #This block gets the path to the hidden text using the unique ID number drawn from the link value
        recordingIDNumber = link.split('/')[-2]
        textTreeName = '//*[@id="audio_request_{0}"]'.format(recordingIDNumber)
        textTree = tree.xpath(textTreeName)

        #This block gets the actual text itself
        text = ''
        for element in textTree[0].getchildren():
            paragraphText = element.text
            if paragraphText != None:#has to check for breaks that would create a type error
                text += paragraphText 
        
        #This block gets the word count total as an integer
        wordCountTree = tree.xpath(nextWordCountTreeName)
        wordCount = wordCountTree[0].text.split()[0] #gets a string of just the number, without ' Words'
        wordCount = int(wordCount)


        #Now, all that's left is to do analysis of the gathered values, determining validity of the recording.  
        RRcount = text.count(SEARCHED_PHONEME)
        validExample =  wordCount > MIN_WORD_COUNT and wordCount < MAX_WORD_COUNT and RRcount >= MIN_RR_OCCURENCE
        if validExample:
            print('RR count:', RRcount) #DEBUG
            validPages[link] = RRcount 

        nextTitleTreeName = '//*[@id="left_panel"]/div/div/div[{0}]/table/tbody/tr/td[2]/div/span[3]/a'.format(str(i))
        nextWordCountTreeName = '//*[@id="left_panel"]/div/div/div[{0}]/table/tbody/tr/td[2]/div[3]/span'.format(str(i))

    return validPages



def readSearchPage(num):
    '''readSearchPage
    This method is built to read a single page of the normal search pages,
    with some sort of search string given. 
    num: The index of the search page to be processed. 
    
    Returns: A dictionary, containing string URLs of valid pages,
    keyed to the ratio of trilled RRs for each.'''

    validPages = {}
    page = requests.get('https://rhinospike.com/search/?page=' + str(num) + '&q=rr&language=2')
    tree = html.fromstring(page.content)

    for i in range(1, 11):
        titleTree =  tree.xpath('//*[@id="left_panel"]/div/div[' + str(i) 
                + ']/table/tbody/tr/td[2]/div[1]/span[3]/a')
        wordCountTree = tree.xpath('//*[@id="left_panel"]/div/div[' + str(i) 
                + ']/table/tbody/tr/td[2]/div[3]/span[1]')

        #Quick and dirty fix for an error where the first element doesn't work properly
        if i == 1:
            titleTree =  tree.xpath('//*[@id="left_panel"]/div/div/table/tbody/tr/td[2]/div/span[3]/a')
            wordCountTree = tree.xpath('//*[@id="left_panel"]/div/div/table/tbody/tr/td[2]/div[3]/span')

        link = WEBSITE_LINK + titleTree[0].get('href')
        title = titleTree[0].text

        #indexing at end gets a string of just the number, without ' Words'
        wordCount = wordCountTree[0].text[:-6]
        wordCount = int(wordCount)

        if wordCount > MIN_WORD_COUNT and wordCount < MAX_WORD_COUNT:
            validPages[title] = link
    return validPages

def getTranscript(link):
    page = requests.get(link)
    tree = html.fromstring(page.content)

    textTree = tree.xpath('//*[@id="left_panel"]/div/div/div/div[2]')
    text = ''
    for element in textTree[0].getchildren():
        text += element.text
    return text

def processPage(link, mod):
    '''
    processPage
    DEPRECIATED, as it fails to properly login to download the files. 
    link: a link to the page to be processed
    session: a session object that's already logged in
    mode: determines which folder to save the results to. 
          Can be 'test', 'training', or 'cv'.
    '''
    page = requests.get(url=link, auth=auth) 
    tree = html.fromstring(page.content)

    textTree = tree.xpath('//*[@id="left_panel"]/div/div/div/div[2]')
    text = ''
    for element in textTree[0].getchildren():
        text += element.text

    if text.count(SEARCHED_PHONEME) < MIN_RR_OCCURENCE:
        return False

    #Fetches the link from the transcription page
    downloadTree = tree.xpath('//*[@id="left_panel"]/div/div/div/ul/li/span[3]/a')

    for element in tree.xpath("//*[@id='left_panel']/div/div/div/ul/li/span[3]/a"):
        print(element.get('href'))
    #If the top link is locked behind a paywall, it skips it. 
    # for child in tree.xpath('//*[@id="left_panel"]/div/div/div/ul')[0].iterchildren():
        # print('Recorder Name:', child.find('/a/strong')[0].text)
        # print('Child:', len(child.find('/span[3]')))
    #while downloadTree.text == 'Unlock' and count < 3:
    #    downloadTree = tree.xpath('//*[@id="left_panel"]/div/div/div/ul/li[2]/span[3]/a')
    #    count += 1
    downloadLink = WEBSITE_LINK + downloadTree[0].get('href')
    

    print('Download Link:', downloadLink) #DEBUG

    # Follows the link to the download page
    # downloadPage = requests.get(downloadLink)
    # downloadPageTree = html.fromstring(page.content)

    #Downloads the actual file
    # downloadLink = downloadPageTree.xpath('//*[@id="body"]/p[4]/a')
    # downloadedFile = requests.get(downloadLink).content
    
    
    #Saves the file with the title of the transcription as its title
    # mp3Name = tree.xpath('//*[@id="left_panel"]/div/div/div/table/tbody/tr/td[2]/div/span[3]/a')[0].text
    # mp3Name = '.\Raw Data\mp3\\' + mode + '\\'+ mp3Name + '.mp3'
    # song = open(mp3Name, 'wb')
    # song.write(downloadedFile)
    # song.close()
    return True

def rhinoSpikeLogin():
    '''
    Pulled from a helpful stackOverflow user: 
    https://stackoverflow.com/questions/8316818/login-to-website-using-python/8316989#8316989

    '''
    # Start a session so we can have persistant cookies
    mySession = requests.Session()

    # This is the form data that the page sends when logging in
    login_data = {
        'loginemail': EMAIL,
        'loginpswd': PASSWORD,
        #'submit': 'Log in',
    }

    # Authenticate
    r = mySession.post(WEBSITE_LINK + '/account/login', data=login_data)
    print('Status Code:', r.status_code) #DEBUG
    
    return mySession

def saveObj(name, obj):
    '''saveObj
    Credit due to user Zah, from:
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file'''
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadObj(name):
    '''loadObj
    Credit due to user Zah, from:
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file'''
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    #finalURLs = loadObj('validURLs')
    finalURLs = {}
    print(finalURLs)
    for num in range(101, 300):
        validPages = readLangPage(num)
        print('Page {0} returned:'.format(num), validPages) #DEBUG
        finalURLs = dict(finalURLs, **validPages)
        print('Len:', len(finalURLs)) #DEBUG
    
    saveObj('validURLs', finalURLs)









#           '//*[@id="left_panel"]/div/div[1]/table/tbody/tr/td[2]/div[3]/span[1]'
#           '//*[@id="left_panel"]/div/div[2]/table/tbody/tr/td[2]/div[3]/span[1]'
#           '//*[@id="left_panel"]/div/div[3]/table/tbody/tr/td[2]/div[3]/span[1]'

    


