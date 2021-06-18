#!/usr/bin/env python3

import matplotlib.pyplot as plt
import requests
import seaborn as sns
from bs4 import BeautifulSoup
import numpy as np
#------

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



while True:
    print("""
    1. Run
    2. Update lottery list (Fetch from EuroJackpot site)
    3. Exit
            """)
    inp = int(input()) 




    if inp == 1:


        try:
            with open("./lottery_numbers.txt", "r") as f:
                lines = f.readlines()

        except IOError:
            print("Lottery file can't be found, try updating the list first" )
            continue 
        

        list = []
        for i in lines:
            list.append(i.replace("\n", "").split(" "))


        for i in range(0, len(list)):
            for j in range(0, len(list[i])):
                list[i][j] = int(list[i][j])

        xa = np.array([]) 
        ya = np.array([])

        # Adding the 5 (five) first numbers and the last two numbers together
        # to make it readable on a scatterplot
        for i in range(0, len(list)): 
            xa = np.append(xa, [sum(list[i][0:5])])
            ya = np.append(ya, [sum(list[i][5:7])])

        #coor = np.reshape(coor, (-1, 2))



        sns.set_theme(style="whitegrid")
        sns.scatterplot(x=xa, y=ya)
        plt.show()

        #print(arr[0:4].sum)

    elif inp == 2:
       updateLotteryList() 

    elif inp == 3:
        break

    else:
        print("not a valid input")

