# DAS
# Centralized neural network implementation using optimal control setup
# slightly different from the code handouts.
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
			a = np.concatenate([a,[[1]]], axis = 0)
			a = w@a
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
			z = np.concatenate([activation,[[1,]*activation.shape[1]]], axis = 0)
			z = w@z
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], YY) * sigmoid_prime(zs[-1])
		nabla_w[-1][:,-1] = np.sum(delta,axis = 1)
		nabla_w[-1][:,:-1] = delta@activations[-2].T

		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = (self.weights[-l+1][:,:-1].T@delta)*sp
			nabla_w[-l][:,-1] = np.sum(delta,axis = 1)
			nabla_w[-l][:,:-1] = delta@activations[-l-1].T
		return nabla_w


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
		Y_hat = np.zeros((n,1))
		Y_true = np.zeros((n,1))
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
			for i in range(n):
				Y_hat[i,0], Y_true[i,0] = self.feedforward(training_data[i][0])[0,0], training_data[i][1][0]
			gradients.append(np.mean(np.abs(self.cost_derivative(Y_hat,Y_true)).flatten()))
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
		XX = np.zeros((mini_batch[0][0].shape[0],len(mini_batch)))
		YY = np.zeros((mini_batch[0][1].shape[0],len(mini_batch)))
		for i, x in enumerate(mini_batch):
			XX[:,i], YY[:,i] = x[0].flatten(), x[1].flatten()
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
			Y_hat[i,0], Y_true[i,0] = self.feedforward(test_data[i][0])[0,0], test_data[i][1][0]
		acc = self.get_accuracy_value(Y_hat, Y_true)
		loss = self.get_cost_value(Y_hat,Y_true)
		return (acc, loss)
		# test_results = [(self.feedforward(x), y)
		# 				for (x, y) in test_data]
		# return (np.mean([self.get_accuracy_value(x,y) for (x, y) in test_results]), 
		# 		np.mean([self.get_cost_value(x,y) for (x,y) in test_results]))


	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
		\partial a for the output activations."""
		return -((y / output_activations) - ((1 - y)/(1 - output_activations)))


	def get_cost_value(self,Y_hat, Y):
		# number of examples
		m = Y_hat.shape[0]
		# calculation of the cost according to the formula
		cost = -1 / m * (Y.T @ np.log(Y_hat) + (1 - Y).T@np.log(1 - Y_hat))
		return np.squeeze(cost)

	def convert_prob_into_class(self,probs):
		probs_ = np.copy(probs)
		probs_[probs_ > 0.5] = 1
		probs_[probs_ <= 0.5] = 0
		return probs_

	def get_accuracy_value(self,Y_hat, Y):
		Y_hat_ = self.convert_prob_into_class(Y_hat)
		return np.sum(Y_hat_.flatten() == Y.flatten())/Y.shape[0]