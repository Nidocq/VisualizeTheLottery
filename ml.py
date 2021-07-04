from numpy.core.fromnumeric import shape
from scipy.sparse import data
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random 

def ml(load_prev=0):
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

	target = [1 for i in range(len(list))]


	def randomLists(a, b, times):
		return [random.randint(a, b) for i in range(times)]

	def createDummyData(size):
		dataSet = np.array(list)
		dataTarget = np.array(target)
		for i in range(size):
			dataSet = np.append(dataSet, np.array([np.append([randomLists(1, 50+1, 5)], randomLists(1, 10+1, 2))]), axis=0)

		dataTarget = np.append(dataTarget, np.zeros(size))
		return dataSet, dataTarget


	dataSet, dataTarget = createDummyData(4000)

	print(f"dataset : {dataSet} with shape of {dataSet.shape} \n dataTarget {dataTarget} with shape as {dataTarget.shape}")
	dfset = pd.DataFrame(dataSet)
	dftarget = pd.DataFrame(dataTarget)

	xtr, xte, ytr, yte = train_test_split(dfset, dftarget)
	fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)

	uni_training = np.array([])
	uni_testing = np.array([])
	dis_training = np.array([])
	dis_testing= np.array([])
	
	epochs = range(1, 20 +1)
	
	
	
	# -- knearest ---
	# Use KNearestNeighbor - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
	
	axs[0, 0].set_title("KNearest Neighbor | Uniform weight")
	axs[1, 0].set_title("KNearest Neighbor | Distance weight")
	for weights in ["uniform", "distance"]:
		for epoch in epochs:
			knn = KNeighborsClassifier(n_neighbors=epoch, weights=weights)
			knn_model = knn.fit(xtr, ytr.values.ravel())
	
			if weights == "uniform":
				uni_training = np.append(knn_model.score(xtr, ytr), uni_training)
				uni_testing = np.append(knn_model.score(xte, yte), uni_testing)
			else:
				dis_training = np.append(knn_model.score(xtr, ytr), dis_training)
				dis_testing = np.append(knn_model.score(xte, yte), dis_testing)
	
	print(f"epochs : {epochs}")
	print(f"uni_testing : {uni_testing} - {uni_testing.shape}")
	
	axs[0, 0].plot(epochs, uni_training, label="Uniform weight training")
	axs[0, 0].plot(epochs, uni_testing, label="Uniform weight testing")
	axs[1, 0].plot(epochs, dis_training, label="Distance weight training")
	axs[1, 0].plot(epochs, dis_testing, label="Distance weight testing")
	
	
	
	# -- Decision tree --
	# Decision tree - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
	
	gini_training = np.array([])
	gini_testing = np.array([])
	entropy_training = np.array([])
	entropy_testing = np.array([])
	
	axs[0, 1].set_title("Decision Tree Classifier | Gini Criteria")
	axs[1, 1].set_title("Decision Tree Classifier | Entropy Criteria")
	for criteria in ["gini", "entropy"]:
		for epoch in epochs:
			dct = DecisionTreeClassifier(criterion=criteria, max_depth=epoch)
			dct_model = dct.fit(xtr, ytr)
	
			if criteria == "gini":
				gini_training = np.append(dct_model.score(xtr, ytr), gini_training)
				gini_testing = np.append(dct_model.score(xte, yte), gini_testing)
			else:
				entropy_training = np.append(dct_model.score(xtr, ytr), entropy_training)
				entropy_testing = np.append(dct_model.score(xte, yte), entropy_testing)
	
	
	axs[0, 1].plot(epochs, gini_training, label="Gini Training")
	axs[0, 1].plot(epochs, gini_testing, label="Gini Testing")
	axs[1, 1].plot(epochs, entropy_training, label="Entropy Training")
	axs[1, 1].plot(epochs, entropy_testing, label="Entropy Testing")
	
	# Random forests - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
	
	gini_training = np.array([])
	gini_testing = np.array([])
	entropy_training = np.array([])
	entropy_testing = np.array([])
	
	axs[0, 2].set_title("Random Forest | Gini Criteria -> epoch 10x")
	axs[1, 2].set_title("Random Forest | Entropy Criteria -> epoch 10x")
	for criteria in ["gini", "entropy"]:
		for epoch in epochs:
			RF = RandomForestClassifier(criterion=criteria, max_depth=(epoch*10)) #Try with n_estimators
			RF_model = RF.fit(xtr, ytr.values.ravel())
	
			if criteria == "gini":
				gini_training = np.append(RF_model.score(xtr, ytr), gini_training)
				gini_testing = np.append(RF_model.score(xte, yte), gini_testing)
			else:
				entropy_training = np.append(RF_model.score(xtr, ytr), entropy_training)
				entropy_testing = np.append(RF_model.score(xte, yte), entropy_testing)
	
	axs[0, 2].plot(epochs, gini_training, label="Gini Training")
	axs[0, 2].plot(epochs, gini_testing, label="Gini Testing")
	axs[1, 2].plot(epochs, entropy_training, label="Entropy Training")
	axs[1, 2].plot(epochs, entropy_testing, label="Entropy Testing")
	
	
	# Neural Network - https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
	
	identity_training = np.array([])
	identity_testing = np.array([])
	logistic_training = np.array([])
	logistic_testing = np.array([])
	tanh_training = np.array([])
	tanh_testing = np.array([])
	relu_training = np.array([])
	relu_testing = np.array([])
	
	
	axs[0, 3].set_title("Neural Network (MLP classifier) | Identity -> 10x ephocs")
	axs[1, 3].set_title("Neural Network | Logistic ")
	axs[2, 3].set_title("Neural Network | Tanh ")
	axs[3, 3].set_title("Neural Network | Relu")
	for activation in ["identity", "logistic", "tanh", "relu"]:
		for epoch in epochs:
			MLP = MLPClassifier(activation=activation, hidden_layer_sizes=(epoch*10))
			MLP_model = MLP.fit(xtr, ytr.values.ravel())
	
			if activation == "identity":
				identity_training = np.append(MLP_model.score(xte, yte), identity_training)
				identity_testing = np.append(MLP_model.score(xtr, ytr), identity_testing)
	
			elif activation == "logistic":
				logistic_training = np.append(MLP_model.score(xtr, ytr), logistic_training)
				logistic_testing = np.append(MLP_model.score(xte, yte), logistic_testing)
	
			elif activation == "tanh":
				tanh_training = np.append(MLP_model.score(xtr, ytr), tanh_training)
				tanh_testing = np.append(MLP_model.score(xte, yte), tanh_testing)
	
			else:
				relu_training = np.append(MLP_model.score(xtr, ytr), relu_training)
				relu_testing = np.append(MLP_model.score(xte, yte), relu_testing)
	
	
	axs[0, 3].plot(epochs, identity_training, label="Identity Training 10x hidn lyrs")
	axs[0, 3].plot(epochs, identity_testing, label="Identity Testing 10x hidn lyrs")
	axs[1, 3].plot(epochs, logistic_training, label="Logistic Testing")
	axs[1, 3].plot(epochs, logistic_testing, label="Logistic Testing")
	axs[2, 3].plot(epochs, tanh_training, label="Tanh Training")
	axs[2, 3].plot(epochs, tanh_testing, label="Tanh Testing")
	axs[3, 3].plot(epochs, relu_training, label="Relu Training")
	axs[3, 3].plot(epochs, relu_testing, label="Relu Testing")
	
	# print(model.score(xtr, ytr))
	# print(model.score(xte, yte))
	# print("prediction probability: {}".format(model.predict_proba([[1, 2, 30, 31, 44, 1, 2]])))
	# print("prediction : {}".format(model.predict([[1, 2, 30, 31, 44, 1, 2]])))
	

	plt.legend()
	plt.ylabel("Accuracy %")
	plt.xlabel("epochs")
	# plt.text(0, 3, f'{testing}', path_effects=[path_effects.Normal()])
	# plt.text(1, 3, 'Hello path effects world!', path_effects=[path_effects.Normal()])
	# plt.text(2, 3, 'Hello path effects world!', path_effects=[path_effects.Normal()])
	# plt.text(3, 3, 'Hello path effects world!', path_effects=[path_effects.Normal()])
	plt.show()

if __name__ == '__main__':
    ml()
else:
    pass

