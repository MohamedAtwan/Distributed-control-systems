#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from mnist_loader import *
from centralized_NN import *
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
	EPOCHS = 1000
	mini_batch_size = 32
	eta = 1e-3
	scaler = MinMaxScaler()

	# Load the data
	x_train, y_train, x_test, y_test = load_data('Data',chosen_class = 8) # chosen class is number 8

	# scale the data
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)

	# some data organization
	train_data = [(np.atleast_2d(x_train[i,:]).T,y_train[i,:]) for i in range(x_train.shape[0])]
	test_data = [(np.atleast_2d(x_test[i,:]).T,y_test[i,:]) for i in range(x_test.shape[0])]

	# Creating the neural network
	D = x_train.shape[1]
	nn = NeuralNetwork([D,32,32,1]) # specifying the number of neurons at each layer in a list as input to the class constructor

	# Run the iterative algorithm
	accuracy_train, accuracy_test, loss_train, loss_test, gradients = nn.SGD(train_data, EPOCHS, mini_batch_size, eta, test_data = test_data)

	# creating plots for accuracy and loss evolution
	plt.figure()
	plt.plot(np.arange(len(accuracy_train)), accuracy_train)
	plt.xlabel(r"Iterations $t$")
	plt.ylabel(r"$Accuracy$")
	plt.title("Evolution of the accuracy metric on train data")
	plt.grid()
	plt.savefig('Figures/Accuracy_Results_train_data_centralized.png')

	plt.figure()
	plt.plot(np.arange(len(loss_train)), loss_train)
	plt.xlabel(r"Iterations $t$")
	plt.ylabel(r"$Loss$")
	plt.title("Evolution of the loss metric on train data")
	plt.grid()
	plt.savefig('Figures/Loss_Results_train_data_centralized.png')

	plt.figure()
	plt.plot(np.arange(len(accuracy_test)), accuracy_test)
	plt.xlabel(r"Iterations $t$")
	plt.ylabel(r"$Accuracy$")
	plt.title("Evolution of the accuracy metric on test data")
	plt.grid()
	plt.savefig('Figures/Accuracy_Results_test_data_centralized.png')

	plt.figure()
	plt.plot(np.arange(len(loss_test)), loss_test)
	plt.xlabel(r"Iterations $t$")
	plt.ylabel(r"$Loss$")
	plt.title("Evolution of the loss metric on test data")
	plt.grid()
	plt.savefig('Figures/Loss_Results_test_data_centralized.png')

	plt.figure()
	plt.semilogy(np.arange(len(gradients)), gradients)
	plt.xlabel(r"Iterations $t$")
	plt.ylabel(r"$Gradients$")
	plt.title("Evolution of the gradients")
	plt.grid()
	plt.savefig('Figures/gradients_centralized.png')
	plt.show()
