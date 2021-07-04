#!/usr/bin/env python3
from updateData import * 
from ml import *
from model import *

while True:
    print("""
    1. Make prediction
    2. Visualize Lottery
    3. Update lottery list (Fetch from EuroJackpot site)
    4. Exit
            """)

    try: 
        inp = int(input("> ")) 

    except ValueError:
        print("Not a valid input")
        continue

    if inp == 1:
        runModel(input("Enter your number to check if it will win: \n e.g -> 1, 3, 33, 43, 46, 1, 10\n "))
        continue

    if inp == 2:
       ml()

    elif inp == 3:
       updateLotteryList() 
       continue

    elif inp == 4:
        break

    else:
        print("not a valid input")

