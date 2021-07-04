import numpy as np
import requests
from bs4 import BeautifulSoup


def updateLotteryList():
    arr = np.array([])

    for i in range(12, 21+1):
        URL = "https://www.euro-jackpot.net/en/results-archive-20" + str(i)
        print(f"Fetching data from: {URL}")
    
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
    
        soup = soup.find("tbody")
        soup = soup.find_all("li")
    
    
        for i in range(0, len(soup), 7):
            for j in range(0, 7):
                arr = np.append(arr, [soup[i+j]])
        
        arr = np.reshape(arr, (-1, 7))
    
        np.savetxt("lottery_numbers.txt", arr, fmt='%s')