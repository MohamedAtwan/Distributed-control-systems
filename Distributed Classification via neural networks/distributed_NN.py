# DAS
# Distributed setup for neural network training
# Authors
# Mohamed Aboraya
# Marco Ghaly
#
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from centralized_NN import *
from sklearn.preprocessing import MinMaxScaler

#### Libraries
# Standard library
import random
from copy import deepcopy

# Third-party libraries
import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy.linalg import block_diag

#### Miscellaneous functions
def sigmoid(z):
	"""The sigmoid function."""
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	"""Derivative of the sigmoid function."""
	return sigmoid(z)*(1-sigmoid(z))

class NeuralNetwork:

	def __init__(self, sizes):
		"""The list ``sizes`` contains the number of neurons in the
		respective layers of the network.  For example, if the list
		was [2, 3, 1] then it would be a three-layer network, with the
		first layer containing 2 neurons, the second layer 3 neurons,
		and the third layer 1 neuron.  The biases and weights for the
		network are initialized randomly, using a Gaussian
		distribution with mean 0, and variance 1.  Note that the first
		layer is assumed to be an input layer, and by convention we
		won't set any biases for those neurons, since biases are only
		ever used in computing the outputs from later layers."""
		self.num_layers = len(sizes)
		self.sizes = sizes
		# self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x)
						for x, y in zip([e+1 for e in sizes[:-1]], sizes[1:])]

	def feedforward(self, a):
		"""Return the output of the network if ``a`` is input."""
		for w in self.weights:
			a = np.concatenate([a,np.array([[1,]*a.shape[0]]).T], axis = 1)
			a = a@w.T
			a = sigmoid(a)
		return a

	def backprop(self,XX, YY):

		nabla_w = [np.zeros(w.shape) for w in self.weights]
		f_ii = []
		# feedforward
		activation = XX
		activations = [XX] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer

		for w in self.weights:
			z = np.concatenate([activation,np.array([[1,]*activation.shape[0]]).T], axis = 1)
			z = z@w.T
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], YY) * sigmoid_prime(zs[-1])
		nabla_w[-1][:,-1] = np.sum(delta,axis = 0)
		nabla_w[-1][:,:-1] = delta.T@activations[-2]

		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			# print(sp.shape)
			# print(delta.shape)
			# print(self.weights[-l+1][:,:-1].shape)
			delta = (delta@self.weights[-l+1][:,:-1])*sp
			nabla_w[-l][:,-1] = np.sum(delta,axis = 0)
			nabla_w[-l][:,:-1] = delta.T@activations[-l-1]
		return nabla_w
# class NeuralNetwork:

# 	def __init__(self, sizes):
# 		"""The list ``sizes`` contains the number of neurons in the
# 		respective layers of the network.  For example, if the list
# 		was [2, 3, 1] then it would be a three-layer network, with the
# 		first layer containing 2 neurons, the second layer 3 neurons,
# 		and the third layer 1 neuron.  The biases and weights for the
# 		network are initialized randomly, using a Gaussian
# 		distribution with mean 0, and variance 1.  Note that the first
# 		layer is assumed to be an input layer, and by convention we
# 		won't set any biases for those neurons, since biases are only
# 		ever used in computing the outputs from later layers."""
# 		self.num_layers = len(sizes)
# 		self.sizes = sizes
# 		# self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
# 		self.weights = [np.random.randn(y,x)
# 						for x, y in zip([e+1 for e in sizes[:-1]], sizes[1:])]

# 	def feedforward(self, a):
# 		"""Return the output of the network if ``a`` is input."""
# 		for u in self.weights:
# 			a = z = np.concatenate([np.array([[1,]*a.shape[0]]).T,a], axis = 1)
# 			a = a@u.T
# 			a = sigmoid(a)
# 		return a

# 	def  adjoint_dynamics(self,ltp,xt,ut):
# 		"""
# 			input: 
# 			          llambda_tp current costate
# 			          xt current state
# 			          ut current input
# 			output: 
# 			          llambda_t next costate
# 			          delta_ut loss gradient wrt u_t
# 		"""
# 		# print()
# 		d_in = ut.shape[1]
# 		d_out = ut.shape[0]
# 		df_dx = np.zeros((d_in-1,d_out))
# 		df_du = np.zeros((d_out*d_in,d_out))
# 		dim = np.tile([d_in],d_out)
# 		cs_idx = np.append(0,np.cumsum(dim))
		
# 		for ell in range(d_out):
# 			temp = np.hstack([1,xt])@np.atleast_2d(ut[ell,:]).T
# 			dsigma_ell = sigmoid_prime(temp)
# 			df_dx[:,ell] = dsigma_ell*ut[ell,1:]
# 			df_du[ cs_idx[ell]:cs_idx[ell+1] , ell] = dsigma_ell*np.hstack([1,xt])

# 		# # temp = np.hstack([np.ones(d_out,1),np.repeat(np.atleast_2d(xt),d_out,axis = 0)])@ut.T
# 		# Xt = np.concatenate([np.array([[1,]*xt.shape[0]]).T,xt], axis = 1)
# 		# # temp = np.hstack([np.ones,xt])@ut.T
# 		# temp = Xt@ut.T
# 		# dsigma_ell = sigmoid_prime(temp)
# 		# # print(Xt.shape)
# 		# # print(dsigma_ell.shape)
# 		# # df_dx = np.repeat(np.atleast_2d(dsigma_ell).T,ut.shape[1]-1,axis = 1)*ut[:,1:]
# 		# df_dx = np.repeat(np.expand_dims(dsigma_ell,-1),ut.shape[1]-1,axis = -1)*np.repeat(np.expand_dims(ut[:,1:],0),dsigma_ell.shape[0],axis = 0)
# 		# # df_du = dsigma_ell[0]*np.hstack([1,xt]).T
# 		# df_du = dsigma_ell[:,0]@Xt
		
# 		# for ell in range(1,d_out):
# 		# 	# df_du = block_diag(df_du,dsigma_ell[ell]*np.hstack([1,xt]).T)
# 		# 	df_du = block_diag(df_du,dsigma_ell[:,ell]@Xt)
# 		# print()
# 		# print(df_du.shape)
# 		# print(ltp.shape)
# 		lt = df_dx@ltp
# 		Delta_ut_vec = df_du@ltp
# 		Delta_ut = np.reshape(Delta_ut_vec,(d_out,d_in))
# 		return lt, Delta_ut

# 	def backprop(self,XX, YY):
# 		N_samples = YY.shape[0]
# 		nabla_w = [np.zeros(w.shape) for w in self.weights]
# 		f_ii = []
# 		# feedforward
# 		activation = XX
# 		activations = [XX] # list to store all the activations, layer by layer
# 		zs = [] # list to store all the z vectors, layer by layer
# 		llambda = [0,]*self.num_layers
# 		Delta_ut = [np.zeros(e.shape) for e in self.weights]
# 		for u in self.weights:
# 			z = np.concatenate([np.array([[1,]*activation.shape[0]]).T,activation], axis = 1)
# 			z = z@u.T
# 			zs.append(z)
# 			activation = sigmoid(z)
# 			activations.append(activation)
# 		llambda[-1] = np.zeros((N_samples,activation.shape[1]))
# 		for i in range(N_samples):
# 			llambda[-1][i,:] = self.cost_derivative(activations[-1][i,:], YY[i,:])
# 		# llambda[-1] = np.sum(self.cost_derivative(activations[-1], YY),axis = 0)
# 		for t in reversed(list(range(self.num_layers-1))):
# 			tmp_du = None
# 			llambda[t] = np.zeros((N_samples,activations[t].shape[1]))
# 			for i in range(N_samples):
# 				llambda[t][i], tmp_du = self.adjoint_dynamics(llambda[t+1][i],activations[t][i,:],self.weights[t])
# 			# llambda[t], Delta_ut[t] = self.adjoint_dynamics(llambda[t+1],activations[t],self.weights[t])
# 				Delta_ut[t] += tmp_du
# 		return Delta_ut

	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
		\partial a for the output activations."""
		return -((y / output_activations) - ((1 - y)/(1 - output_activations)))


	def get_cost_value(self,Y_hat, Y):
		# number of examples
		m = Y_hat.shape[0]
		# calculation of the cost according to the formula
		cost = (-1 / m )* (Y.T @ np.log(Y_hat) + (1 - Y).T@np.log(1 - Y_hat))
		return np.squeeze(cost)

	def convert_prob_into_class(self,probs):
		probs_ = np.copy(probs)
		probs_[probs_ > 0.5] = 1
		probs_[probs_ <= 0.5] = 0
		return probs_

	def get_accuracy_value(self,Y_hat, Y):
		Y_hat_ = self.convert_prob_into_class(Y_hat)
		return np.sum(Y_hat_.flatten() == Y.flatten())/Y.shape[0]


# class Agent:
# 	def __init__(self,sizes, data):
# 		self.x_train, self.y_train, self.x_test, self.y_test = data
# 		self.nn = NeuralNetwork(sizes)
# 		self.XX = self.nn.weights
# 		self.YY = self.nn.backprop(self.x_train, self.y_train)

# 	def get_state(self):
# 		return self.XX

# 	def get_gradient(self):
# 		return self.YY

# 	def update(self,idx,WW,ss,Nii, agents):
# 		XXp = [None,]*len(self.XX)
# 		n_batch = self.x_train.shape[0]
# 		YY_temp = deepcopy(self.YY)

# 		# *********step one and two @ slide 30 of learning with neural network *********
# 		for k, xx in enumerate(self.XX): 
# 			XXp[k] = WW[idx, idx]*xx - (ss/n_batch)*self.YY[k] #######
# 		for jj in Nii:
# 			XX_neighbour = agents[jj].get_state()
# 			for k in range(len(XXp)):
# 				XXp[k] += WW[idx,jj]* XX_neighbour[k]

# 		# ******** step three @ slide 30 of learning with neural network ***********
# 		f_ii = self.nn.get_cost_value(self.nn.feedforward(self.x_train), self.y_train)
# 		grad_fii = deepcopy(self.YY)
# 		self.XX = deepcopy(XXp)
# 		# self.nn.weights = deepcopy(XXp)
# 		# grad_fii_p = self.nn.backprop(self.x_train, self.y_train)
# 		# for k in range(len(self.YY)):
# 		# 	self.YY[k] = WW[idx, idx] * self.YY[k] + (grad_fii_p[k] - grad_fii[k])
# 		# 	YY_neighbour = agents[jj].get_gradient()
# 		# 	for jj in Nii:
# 		# 		self.YY[k] += WW[idx,jj]* YY_neighbour[k]

			
# 		# *******updating the local estimate ***********************
# 		self.nn.weights = deepcopy(self.XX)
# 		grad_fii_p = self.nn.backprop(self.x_train, self.y_train)
# 		for k in range(len(self.YY)):
# 			self.YY[k] = WW[idx, idx] * self.YY[k] + (grad_fii_p[k] - grad_fii[k])
		
# 		for jj in Nii:
# 			YY_neighbour = agents[jj].get_gradient()
# 			for k in range(len(YY_neighbour)):
# 				self.YY[k] += WW[idx,jj]* YY_neighbour[k]
# 		return self.XX, self.YY, f_ii

# 	def cost_derivative(self, output_activations, y):
# 		"""Return the vector of partial derivatives \partial C_x /
# 		\partial a for the output activations."""
# 		return -((y / output_activations) - ((1 - y)/(1 - output_activations)))


# 	def get_cost_value(self,Y_hat, Y):
# 		# number of examples
# 		m = Y_hat.shape[0]
# 		# calculation of the cost according to the formula
# 		cost = (-1 / m )* (Y.T @ np.log(Y_hat) + (1 - Y).T@np.log(1 - Y_hat))
# 		return np.squeeze(cost)

# 	def convert_prob_into_class(self,probs):
# 		probs_ = np.copy(probs)
# 		probs_[probs_ > 0.5] = 1
# 		probs_[probs_ <= 0.5] = 0
# 		return probs_

# 	def get_accuracy_value(self,Y_hat, Y):
# 		Y_hat_ = self.convert_prob_into_class(Y_hat)
# 		return np.sum(Y_hat_.flatten() == Y.flatten())/Y.shape[0]


class Agent:
	def __init__(self,sizes, data, batch_size = 32):
		self.x_train, self.y_train, self.x_test, self.y_test = data
		self.nn = NeuralNetwork(sizes)
		self.XX = self.nn.weights
		self.batch_size = batch_size
		for i in range(self.batch_size):
			if i == 0:
				self.YY = self.nn.backprop(self.x_train[i*self.batch_size:(i+1)*self.batch_size], self.y_train[i*self.batch_size:(i+1)*self.batch_size])
			else:
				tmp = self.nn.backprop(self.x_train[i*self.batch_size:(i+1)*self.batch_size], self.y_train[i*self.batch_size:(i+1)*self.batch_size])
				for k in range(len(tmp)):
					self.YY[k] += tmp[k]
		

	def get_state(self):
		return self.XX

	def get_gradient(self):
		return self.YY

	def get_cost_gradient(self):
		y_hat = self.nn.feedforward(self.x_train)

		self.nn.get_cost_value(self.nn.feedforward(self.x_train), self.y_train)

	# def update(self,idx,WW,ss,Nii, agents):
	# 	XXp = [None,]*len(self.XX)
	# 	n_batch = self.x_train.shape[0]
	# 	YY_temp = deepcopy(self.YY)

	# 	# *********step one and two @ slide 30 of learning with neural network *********
	# 	for k, xx in enumerate(self.XX): 
	# 		XXp[k] = WW[idx, idx]*xx - (ss/n_batch)*self.YY[k] #######
	# 	for jj in Nii:
	# 		XX_neighbour = agents[jj].get_state()
	# 		for k in range(len(XXp)):
	# 			XXp[k] += WW[idx,jj]* XX_neighbour[k]

	# 	# ******** step three @ slide 30 of learning with neural network ***********
	# 	f_ii = self.nn.get_cost_value(self.nn.feedforward(self.x_train), self.y_train)
	# 	grad_fii = deepcopy(self.YY)
	# 	self.XX = deepcopy(XXp)
	# 	# self.nn.weights = deepcopy(XXp)
	# 	# grad_fii_p = self.nn.backprop(self.x_train, self.y_train)
	# 	# for k in range(len(self.YY)):
	# 	# 	self.YY[k] = WW[idx, idx] * self.YY[k] + (grad_fii_p[k] - grad_fii[k])
	# 	# 	YY_neighbour = agents[jj].get_gradient()
	# 	# 	for jj in Nii:
	# 	# 		self.YY[k] += WW[idx,jj]* YY_neighbour[k]

			
	# 	# *******updating the local estimate ***********************
	# 	self.nn.weights = deepcopy(self.XX)
	# 	grad_fii_p = []
	# 	for i in range(self.batch_size):
	# 		tmp = self.nn.backprop(self.x_train[i*self.batch_size:(i+1)*self.batch_size], self.y_train[i*self.batch_size:(i+1)*self.batch_size])
	# 		if i == 0:
	# 			for k in range(len(tmp)):
	# 				grad_fii_p.append(tmp[k])
	# 		else:
	# 			grad_fii_p[k] += tmp[k]

	# 	for k in range(len(self.YY)):
	# 		self.YY[k] = WW[idx, idx] * self.YY[k] + (grad_fii_p[k] - grad_fii[k])
		
	# 	for jj in Nii:
	# 		YY_neighbour = agents[jj].get_gradient()
	# 		for k in range(len(YY_neighbour)):
	# 			self.YY[k] += WW[idx,jj]* YY_neighbour[k]
	# 	return self.XX, self.YY, f_ii

	def update(self,idx,WW,ss,Nii, agents):
		XXp = [None,]*len(self.XX)
		n_batch = self.x_train.shape[0]
		YY_temp = deepcopy(self.YY)

		# *********step one and two @ slide 30 of learning with neural network *********
		for k, xx in enumerate(self.XX):
			XXp[k] = WW[idx, idx]*xx - (ss/self.batch_size)*self.YY[k]
		for jj in Nii:
			XX_neighbour = agents[jj].get_state()
			for k in range(len(XXp)):
				XXp[k] += WW[idx,jj]* XX_neighbour[k]

		# ******** step three @ slide 30 of learning with neural network ***********
		f_ii = self.nn.get_cost_value(self.nn.feedforward(self.x_train), self.y_train)
		grad_fii = deepcopy(self.YY)
		self.XX = deepcopy(XXp)
		self.nn.weights = deepcopy(XXp)
		# grad_fii_p = self.nn.backprop(self.x_train, self.y_train)
		grad_fii_p = []
		for i in range(self.batch_size):
			tmp = self.nn.backprop(self.x_train[i*self.batch_size:(i+1)*self.batch_size], self.y_train[i*self.batch_size:(i+1)*self.batch_size])
			if i == 0:
				for k in range(len(tmp)):
					grad_fii_p.append(tmp[k])
			else:
				grad_fii_p[k] += tmp[k]
		# for k, xx in enumerate(self.XX):
		# 	self.XX[k] -= ss*grad_fii_p[k]

			
		# *******updating the local estimate ***********************
		# self.nn.weights = deepcopy(self.XX)
		# grad_fii_p = self.nn.backprop(self.x_train, self.y_train)
		for k in range(len(self.YY)):
			self.YY[k] = WW[idx, idx] * self.YY[k] + (grad_fii_p[k] - grad_fii[k])
		
		for jj in Nii:
			YY_neighbour = agents[jj].get_gradient()
			for k in range(len(YY_neighbour)):
				self.YY[k] += WW[idx,jj]* YY_neighbour[k]
		return self.XX, self.YY, f_ii


class GradientTracking:
	def __init__(self,data,sizes, NN, MAXITERS, lr = 1e-3):
		p = 0.5
		self.lr = lr
		self.XX = {i:[] for i in range(MAXITERS)}
		self.YY = {i:[] for i in range(MAXITERS)}
		self.FF = np.zeros((MAXITERS))
		self.FF_grad = np.zeros((MAXITERS))
		self.NN, self.max_iters = NN, MAXITERS
		self.G, self.Adj, self.E, self.Deg, self.WW = self.create_graph(p_ER = p)
		self.agents = self.create_agents(NN,data,sizes)
		self.W_agents = np.zeros((NN,3,MAXITERS))

	def create_graph(self,p_ER = 0.3):
		NN = self.NN
		I_NN = np.eye(NN)

		while 1:
			# G = nx.binomial_graph(self.NN, p_ER)
			G = nx.cycle_graph(self.NN)
			Adj = nx.adjacency_matrix(G).toarray()
			E = [e for e in G.edges]
			Deg = Adj@np.ones((Adj.shape[0],1))
			Deg = np.diag(Deg.flatten())
			WW = np.zeros(Adj.shape)
			for ii in range(Adj.shape[0]):
				for jj in range(Adj.shape[1]):
					if (ii,jj) in E and not(ii==jj):
						WW[ii,jj] = 1/(1+max(Deg[ii,ii], Deg[jj,jj]))
					elif (jj, ii) in E and not(ii==jj):
						WW[ii,jj] = 1/(1+max(Deg[ii,ii], Deg[jj,jj]))
					else:
						WW[ii,jj] = 0.0
			for ii in range(WW.shape[0]):
				Nii = np.nonzero(Adj[ii])[0]
				WW[ii,ii] = 1-np.sum([WW[ii,hh] if not(ii==hh) else 0.0 for hh in Nii])

			# print('Check Stochasticity:\n row: {} \n column {}'.format(np.sum(WW,axis=1),np.sum(WW,axis=0)))
			test = np.linalg.matrix_power((I_NN+Adj),NN)
			
			if np.all(test>0):
				print("the graph is connected\n")
				break 
			else:
				print("the graph is NOT connected\n")
				# quit()
		return G, Adj, E, Deg, WW

	def create_agents(self,NN,data,sizes):
		agents = []
		for i in range(NN):
			agents.append(Agent(sizes, data[i]))
			self.XX[0].append(agents[-1].get_state())
			self.YY[0].append(agents[-1].get_gradient())
		return agents

	def update(self,iter):
		NN = self.NN
		for ii in range(NN):
			self.W_agents[ii,0,iter] = self.XX[iter][ii][-1][0,0]
			self.W_agents[ii,1,iter] = self.XX[iter][ii][-2][0,0]
			self.W_agents[ii,2,iter] = self.XX[iter][ii][-3][0,0]
			if iter == self.max_iters-1:
				Nii = np.nonzero(self.Adj[ii])[0]
				_, yy, f_ii = self.agents[ii].update(ii,self.WW,self.lr,Nii,self.agents)
				self.FF[-1] += f_ii
				Y_hat, Y_true = self.agents[ii].nn.feedforward(self.agents[ii].x_train), self.agents[ii].y_train
				self.FF_grad[iter] += np.mean(np.abs(self.agents[ii].nn.cost_derivative(Y_hat,Y_true)))
				# for k in range(self.agents[ii].nn.num_layers-1):

				# 	self.FF_grad[iter] += np.linalg.norm(yy[k])
			else:
				Nii = np.nonzero(self.Adj[ii])[0]
				y = self.agents[ii].update(ii,self.WW,self.lr,Nii,self.agents)
				self.XX[iter+1].append(y[0])
				self.YY[iter+1].append(y[1])
				f_ii = y[2]
				self.FF[iter] += f_ii
				Y_hat, Y_true = self.agents[ii].nn.feedforward(self.agents[ii].x_train), self.agents[ii].y_train
				self.FF_grad[iter] += np.mean(np.abs(self.agents[ii].nn.cost_derivative(Y_hat,Y_true)))

				# for k in range(self.agents[ii].nn.num_layers-1):
				# 	self.FF_grad[iter] += np.linalg.norm(self.YY[iter+1][-1][k])

	def get_metrics(self):
		Y_hat = self.agents[0].nn.feedforward(self.agents[0].x_train)
		Y = self.agents[0].y_train
		cost = self.agents[0].nn.get_cost_value(Y_hat, Y)
		acc = self.agents[0].nn.get_accuracy_value(Y_hat, Y)

		Y_hat = self.agents[0].nn.feedforward(self.agents[0].x_test)
		Y = self.agents[0].y_test
		cost_test = self.agents[0].nn.get_cost_value(Y_hat, Y)
		acc_test = self.agents[0].nn.get_accuracy_value(Y_hat, Y)
		return cost, acc, cost_test, acc_test

	def plotCostEvolution(self):
		# plt.figure()
		# plt.semilogy(np.arange(self.max_iters), self.FF)
		# plt.xlabel(r"iterations $t$")
		# plt.ylabel(r"$x_{i,t}$")
		# plt.title("Evolution of the cost")
		# plt.grid()
		layer_name = None
		for j in range(3):
			if j == 0:
				layer_name = "last layer"
			elif j == 1:
				layer_name = "layer befor the last one"
			else:
				layer_name = "middle layer"
			plt.figure()
			for i in range(self.NN):
				plt.plot(np.arange(self.max_iters),self.W_agents[i,j,:])
			plt.xlabel("iterations")
			plt.ylabel(f"weight value")
			plt.title(f'weights Evolution of neurons at {layer_name}')
			plt.savefig(f"weights_evolution_linear_scale_{layer_name}.png")

		# for j in range(3):
		# 	if j == 0:
		# 		layer_name = "last layer"
		# 	elif j == 1:
		# 		layer_name = "layer befor the last one"
		# 	else:
		# 		layer_name = "middle layer"
		# 	plt.figure()
		# 	for i in range(self.NN):
		# 		plt.semilogy(np.arange(self.max_iters),self.W_agents[i,j,:])
		# 	plt.xlabel("iterations")
		# 	plt.ylabel(f"a weight value of a neuron")
		# 	plt.title(f'weights Evolution of neurons at {layer_name} Logarithmic scale')
		# 	plt.savefig(f"weights_evolution_logarithmic_scale_{layer_name}.png")
		plt.figure()
		plt.semilogy(np.arange(self.max_iters), self.FF_grad)
		plt.xlabel(r"iterations $t$")
		plt.ylabel(r"$Gradient$")
		plt.title("Evolution of the gradient of the cost")
		plt.grid()
		plt.savefig('Figures/Gradient_evolution.png')
		plt.show()

