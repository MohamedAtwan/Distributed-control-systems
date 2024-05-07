#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from mnist_loader import *
from centralized_NN_optcon import *
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from distributed_NN import *
import networkx as nx

if __name__ == '__main__':
	N_agents = 10
	mini_batch_size = 32
	eta = 1e-3
	MAXITERS = np.int(10000) # Explicit Casting

	# Load the data
	data = load_data_distributed('Data',N_agents,chosen_class = 8)

	# initialize the gradient tracking instance
	D = data[0][0].shape[1]
	sizes = [D,32,32,1]
	GT = GradientTracking(data,sizes, N_agents, MAXITERS, lr = 1e-3)

	# storage for the results
	accuracy_results = []
	cost_results = []
	accuracy_test_results = []
	cost_test_results = []

	# Iterate to update and reach consensus
	for iter in tqdm(range(MAXITERS)):
		if (iter % 1) == 0:
			cost, acc, cost_test, acc_test = GT.get_metrics()
			accuracy_results.append(acc)
			cost_results.append(cost)
			accuracy_test_results.append(acc_test)
			cost_test_results.append(cost_test)
			tqdm.write(f"Iteration: {iter}, cost: {cost:.4f}, accuracy:{acc:.4f}, cost_test: {cost_test:.4f}, accuracy_test: {acc_test:.4f}")
		GT.update(iter)
	
	# Plotting the gradients
	GT.plotCostEvolution()

	# More plots for the accuracy and loss evolution
	plt.figure()
	plt.plot(np.arange(len(accuracy_results)), accuracy_test_results)
	plt.xlabel(r"Iterations $t$")
	plt.ylabel(r"$Accuracy_{i,t}$")
	plt.title("Evolution of the accuracy metric on test data")
	plt.grid()
	plt.savefig('Figures/Accuracy_Results_test_data.png')

	plt.figure()
	plt.plot(np.arange(len(accuracy_results)), accuracy_results)
	plt.xlabel(r"Iterations $t$")
	plt.ylabel(r"$Accuracy_{i,t}$")
	plt.title("Evolution of the accuracy metric on train data")
	plt.grid()
	plt.savefig('Figures/Accuracy_Results_train_data.png')

	plt.figure()
	plt.plot(np.arange(len(cost_test_results)), cost_test_results)
	plt.xlabel(r"Iterations $t$")
	plt.ylabel(r"$Cost$")
	plt.title("Evolution of the loss metric on test data")
	plt.grid()
	plt.savefig('Figures/BCE_Results_test_data.png')

	plt.figure()
	plt.plot(np.arange(len(cost_results)), cost_results)
	plt.xlabel(r"Iterations $t$")
	plt.ylabel(r"$Cost$")
	plt.title("Evolution of the BCE on train data")
	plt.grid()
	plt.savefig('Figures/BCE_Results_train_data.png')
	plt.show()




