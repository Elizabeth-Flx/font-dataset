from bs4 import BeautifulSoup
from PIL import Image, ImageOps
import requests
import wget
import os

''' This is used to download fonts from dafs.com'''

headers = {'User-Agent': 'Mozilla/5.0'}

downloadURLs = []
errors = []

def getDownloadURLs(url):
    '''Gets all download URL's from given page and adds the to list (downloadURLs).'''
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, features='html.parser')

    for a in soup.find_all('a', href = True):
        #* Add all download URL's to list (downloadURLs)
        if ("//dl.daf.com/dl/?f=" in a['href']):
            print("Found download URL:", a['href'])
            downloadURLs.append(a['href'])

def getAllFonts(url, nPages):
    '''Takes dafs URL and goes through n ammount of pages.
    All download URL's are added to list (downloadURLs).'''
    for i in range(nPages):
        print("Getting download files from page " + str(i+1) + ':')
        getDownloadURLs(url + '&page=' + str(i+1))

def downloadFonts(targetDir):
    '''Downloads fonts using download URL's in list (downloadURLs) to directory (targetDir).'''

    finished = os.listdir(targetDir)

    for i in downloadURLs:
        if i.split('=')[1] + '.zip' not in finished:
            print("Downloading: " + i.split('=')[1])
            try:
                wget.download('https://' + i, targetDir)
            except Exception as e:
                print("Error occured when downloading: " + i)
                print(str(e))
                errors.append(i)

            print("")
    print("Done!")
