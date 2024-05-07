# Data loader
# Author: Mohamed Aboraya
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os
from sklearn.preprocessing import MinMaxScaler

def prepare_data(train_csv, chosen_class):
	x_train_list = [train_csv.values[train_csv['label'].values==chosen_class,1:]]
	y_train_list = [np.ones((x_train_list[-1].shape[0],1))]
	NN = x_train_list[0].shape[0]
	# x_train_list[0] = x_train_list[0][:NN//4]
	# y_train_list[0] = y_train_list[0][:NN//4]


	Nclasses = len(np.unique(train_csv['label'].values))
	Nchosen = x_train_list[0].shape[0]
	for i in range(Nclasses):
		if i == chosen_class:
			continue
		idx = np.random.choice(list(train_csv.index[train_csv['label']==i]),size = Nchosen//9)
		x_train_list.append(train_csv.values[idx,1:])
		y_train_list.append(np.zeros((x_train_list[-1].shape[0],1)))

	x_train = np.concatenate(x_train_list,axis = 0)
	y_train = np.concatenate(y_train_list,axis = 0)
	idx = np.arange(x_train.shape[0])
	np.random.shuffle(idx)
	x_train = x_train[idx,:]
	y_train = y_train[idx,:]


	return x_train, y_train

def load_data(path, chosen_class = 0):

	# load csv files for both train and test data
	train_csv = pd.read_csv(os.path.join(path,'fashion-mnist_train.csv'))
	test_csv = pd.read_csv(os.path.join(path,'fashion-mnist_test.csv'))

	x_train, y_train = prepare_data(train_csv, chosen_class)
	x_test, y_test = prepare_data(test_csv, chosen_class)
	return x_train, y_train, x_test, y_test

def load_data_distributed(path,N, chosen_class = 0):
	scaler = MinMaxScaler()
	# load csv files for both train and test data
	train_csv = pd.read_csv(os.path.join(path,'fashion-mnist_train.csv'))
	test_csv = pd.read_csv(os.path.join(path,'fashion-mnist_test.csv'))

	x_train, y_train = prepare_data(train_csv, chosen_class)
	x_test, y_test = prepare_data(test_csv, chosen_class)
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)
	data = []
	n_train = x_train.shape[0]//N
	n_test = x_test.shape[0]//N
	for i in range(N):
		data.append((x_train[i*n_train:(i+1)*n_train], y_train[i*n_train:(i+1)*n_train],
							x_test, y_test))
	# for i in range(N):
	# 	data.append((x_train[i*n_train:(i+1)*n_train], y_train[i*n_train:(i+1)*n_train],
	# 						x_test[i*n_test:(i+1)*n_test], y_test[i*n_test:(i+1)*n_test]))
	return data








