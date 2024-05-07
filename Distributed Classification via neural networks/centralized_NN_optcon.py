# DAS
# Centralized neural network implementation using optimal control setup
# as in the course handouts
# Authors
# Mohamed Aboraya
# Marco Ghaly
#
#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
from tqdm import tqdm
from scipy.linalg import block_diag

#### Miscellaneous functions
def sigmoid(z):
	"""The sigmoid function."""
	# return 1.0/(1.0+np.exp(-z))
	return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

def sigmoid_prime(z):
	"""Derivative of the sigmoid function."""
	# return sigmoid(z)*(1-sigmoid(z))
	return (1 - (sigmoid(np.exp2(2) * z)))

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
		self.weights = [np.random.randn(y,x)
						for x, y in zip([e+1 for e in sizes[:-1]], sizes[1:])]

	def feedforward(self, a):
		"""Return the output of the network if ``a`` is input."""
		for u in self.weights:
			a = z = np.concatenate([np.array([[1,]*a.shape[0]]).T,a], axis = 1)
			a = a@u.T
			a = sigmoid(a)
		return a

	def  adjoint_dynamics(self,ltp,xt,ut):
		"""
			input: 
			          llambda_tp current costate
			          xt current state
			          ut current input
			output: 
			          llambda_t next costate
			          delta_ut loss gradient wrt u_t
		"""
		d_in = ut.shape[1]
		d_out = ut.shape[0]
		df_dx = np.zeros((d_in-1,d_out))
		df_du = np.zeros((d_out*d_in,d_out))
		dim = np.tile([d_in],d_out)
		cs_idx = np.append(0,np.cumsum(dim))
		# print(df_du.shape)
		for ell in range(d_out):
			temp = np.hstack([1,xt])@np.atleast_2d(ut[ell,:]).T
			dsigma_ell = sigmoid_prime(temp)
			df_dx[:,ell] = dsigma_ell*ut[ell,1:]
			df_du[ cs_idx[ell]:cs_idx[ell+1] , ell] = dsigma_ell*np.hstack([1,xt])
		lt = df_dx@ltp
		Delta_ut_vec = df_du@ltp
		Delta_ut = np.reshape(Delta_ut_vec,(d_out,d_in))
		return lt, Delta_ut

	def backprop(self,XX, YY):
		"""
			This function calculates the backpropagation of the gradients
			Inputs: XX the images
					YY the labels
			outputs: Delta_ut the gradient direction for all the weights in the network
		"""
		N_samples = YY.shape[0]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		f_ii = []
		# feedforward
		activation = XX
		activations = [XX] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		llambda = [0,]*self.num_layers
		Delta_ut = [np.zeros(e.shape) for e in self.weights]
		for u in self.weights:
			z = np.concatenate([np.array([[1,]*activation.shape[0]]).T,activation], axis = 1)
			z = z@u.T
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		llambda[-1] = np.zeros((N_samples,activation.shape[1]))
		for i in range(N_samples):
			llambda[-1][i,:] = self.cost_derivative(activations[-1][i,:], YY[i,:])
		# llambda[-1] = np.sum(self.cost_derivative(activations[-1], YY),axis = 0)
		for t in reversed(list(range(self.num_layers-1))):
			tmp_du = None
			llambda[t] = np.zeros((N_samples,activations[t].shape[1]))
			for i in range(N_samples):
				llambda[t][i], tmp_du = self.adjoint_dynamics(llambda[t+1][i],activations[t][i,:],self.weights[t])
				Delta_ut[t] += tmp_du
		return Delta_ut

	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		"""Train the neural network using mini-batch stochastic
		gradient descent.  The ``training_data`` is a list of tuples
		``(x, y)`` representing the training inputs and the desired
		outputs.  The other non-optional parameters are
		self-explanatory.  If ``test_data`` is provided then the
		network will be evaluated against the test data after each
		epoch, and partial progress printed out.  This is useful for
		tracking progress, but slows things down substantially."""
		if test_data: 
			n_test = len(test_data)
		n = len(training_data)
		accuracy_train, accuracy_test, losses_train, losses_test, gradients = [], [], [], [], []
		for j in tqdm(range(epochs)):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
			deltas = []
			for mini_batch in mini_batches:
				delta = self.update_mini_batch(mini_batch, eta)
				deltas.append(delta[-1])
			acc_train, loss_train = self.evaluate(training_data)
			acc, loss = self.evaluate(test_data)
			accuracy_train.append(acc_train)
			accuracy_test.append(acc)
			losses_train.append(loss_train)
			losses_test.append(loss)
			gradients.append(np.mean(deltas))
			tqdm.write(f"Epoch: {j}, loss: {loss_train:.4f}, accuracy: {acc_train:.4f}, loss_test: {loss:.4f}, acc_test: {acc:.4f}")
		return accuracy_train, accuracy_test, losses_train, losses_test, gradients
			# else:
			# 	tqdm.write(f"Epoch {j} complete")

	def update_mini_batch(self, mini_batch, eta):
		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
		is the learning rate."""
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		XX = np.zeros((len(mini_batch),mini_batch[0][0].shape[0]))
		YY = np.zeros((len(mini_batch),mini_batch[0][1].shape[0]))
		for i, x in enumerate(mini_batch):
			XX[i,:], YY[i,:] = x[0].flatten(), x[1].flatten()
		delta_nabla_w = self.backprop(XX, YY)
		nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
		return nabla_w


	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		N = len(test_data)
		Y_hat = np.zeros((N,1))
		Y_true = np.zeros((N,1))
		for i in range(N):
			Y_hat[i,0], Y_true[i,0] = self.feedforward(test_data[i][0].T)[0,0], test_data[i][1][0]
		acc = self.get_accuracy_value(Y_hat, Y_true)
		loss = self.get_cost_value(Y_hat,Y_true)
		return (acc, loss)


	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
		\partial a for the output activations."""
		return -((y / output_activations) - ((1 - y)/(1 - output_activations)))


	def get_cost_value(self,Y_hat, Y):
		# number of examples
		# calculation of the cost according to the formula
		m = Y_hat.shape[0]
		cost = -1 / m * (Y.T @ np.log(Y_hat) + (1 - Y).T@np.log(1 - Y_hat))
		return cost[0,0]

	def convert_prob_into_class(self,probs):
		probs_ = np.copy(probs)
		probs_[probs_ > 0.5] = 1
		probs_[probs_ <= 0.5] = 0
		return probs_

	def get_accuracy_value(self,Y_hat, Y):
		Y_hat_ = self.convert_prob_into_class(Y_hat)
		return np.sum(Y_hat_.flatten() == Y.flatten())/Y.shape[0]
