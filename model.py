from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import pandas as pd
import numpy as np
import os.path
import random
import os


def randomLists(a, b, times):
	return [random.randint(a, b) for i in range(times)]

def createDummyData(size, list):
	target = [1 for i in range(len(list))]
	dataSet = np.array(list)
	dataTarget = np.array(target)
	for i in range(size):
		dataSet = np.append(dataSet, np.array([np.append([randomLists(1, 50+1, 5)], randomLists(1, 10+1, 2))]), axis=0)

	dataTarget = np.append(dataTarget, np.zeros(size))
	return dataSet, dataTarget

def runModel(lottery_number):
	try:
		with open("./lottery_numbers.txt", "r") as f:
			lines = f.readlines()
	
	except IOError:
		print("Lottery file can't be found, try updating the list first" )
				

	list = []
	for i in lines:
		list.append(i.replace("\n", "").split(" "))
	
	
	for i in range(0, len(list)):
		for j in range(0, len(list[i])):
			list[i][j] = int(list[i][j])


	dataSet, dataTarget = createDummyData(4000, list)

	dfset = pd.DataFrame(dataSet)
	dftarget = pd.DataFrame(dataTarget)

	xtr, xte, ytr, yte = train_test_split(dfset, dftarget)
	MLP = MLPClassifier(activation="relu", hidden_layer_sizes=70)

	lottery_number = lottery_number.split(",")
	for value in range(0, len(lottery_number)):
		lottery_number[value] = int(lottery_number[value])

	if not os.path.isfile("model.joblib"):
		MLP_model = MLP.fit(xtr, ytr.values.ravel())

		dump(MLP_model, "model.joblib")

		

		try:
			print(f"Model testscore is : {MLP_model.score(xte, yte)}")
			print(f"Model Prediction : {MLP_model.predict([lottery_number])}")
			print(f"Model Porbability : {MLP_model.predict_proba([lottery_number])}")
		except ValueError:
			print("Please provide 7 arguments")

	else:
		print("Using previously trained model ...")
		MLP_model = load("model.joblib")

		try:
			print(f"Model testscore is : {MLP_model.score(xte, yte)}")
			print(f"Model Prediction : {MLP_model.predict([lottery_number])}")
			print(f"Model Porbability : {MLP_model.predict_proba([lottery_number])}")
		except ValueError:
			print("Please provide 7 arguments")