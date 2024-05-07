#
# Gradient Tracking victorized Case
# Mohamed Aboraya
# Marco Ghaly
#
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy
# np.random.seed(0)
from tqdm import tqdm

class Agent:
	# Agent class
	def __init__(self, Q, R, dim = 10):
		# initialize the agent parameters
		self.Q = Q 
		self.R = R 

		self.XX = 2*np.random.rand(dim,1)
		_, self.YY = self.quadratic_fn()

	def get_optcost_value(self):
		""" This function computes the optimal value of the cost for the agent
		"""
		R = self.R 
		Q = self.Q 
		XX  = -np.linalg.inv(Q)@R 
		fval = 0.5*XX.T@(Q@XX)+R.T@XX
		return fval

	def get_state(self):
		# returns the state of the agent
		return self.XX

	def get_gradient(self):
		# return the gradients of the agent
		return self.YY

	def update(self,idx,WW,ss,Nii, agents):
		"""
				updates the state of the agent based on gradient tracking method
				idx: the agent index
				WW: the weighted adjaceny matrix
				ss: step size
				Nii: a list of all the number of agent ii
				agents: a list of all the agents
		"""
		XXp = WW[idx, idx]*self.XX - ss*self.YY
		for jj in Nii:
			XXp += WW[idx,jj]* agents[jj].get_state()
		f_ii, grad_fii = self.quadratic_fn()
		self.XX = XXp.copy()
		_, grad_fii_p = self.quadratic_fn()
		self.YY = WW[idx, idx] * self.YY + (grad_fii_p - grad_fii)
		for jj in Nii:
			self.YY += WW[idx,jj]* agents[jj].get_gradient()
		return self.XX, self.YY, f_ii


	def quadratic_fn(self):
		XX = self.XX
		Q = self.Q 
		R = self.R
		fval = 0.5*XX.T@(Q@XX)+R.T@XX
		fgrad = Q@XX+R
		return fval, fgrad


class GradientTracking:
	# gradient tracking main class
	def __init__(self,agent_dim, NN, MAXITERS, lr = 1e-3):
		# the parameters of the algorithm
		p = 0.3 # probability of the graph
		self.lr = lr # learning rate
		self.XX = np.zeros((agent_dim,NN,MAXITERS)) # the states of the agents
		self.YY = np.zeros((agent_dim,NN,MAXITERS)) # the gradients of the agents
		self.FF = np.zeros((MAXITERS)) # The cost function
		self.FF_grad = np.zeros((MAXITERS)) # The gradient of the cost function
		self.agent_dim = agent_dim # the dimension of the agent
		self.NN, self.max_iters = NN, MAXITERS # number of agents, and number of maximum iterations
		self.G, self.Adj, self.E, self.Deg, self.WW = self.create_graph(p_ER = p) # creating a connected graph
		self.agents = self.create_agents(NN) # creating NN of agents


	def create_graph(self,p_ER = 0.3):
		I_NN = np.eye(self.NN)
		while 1:
			G = nx.binomial_graph(self.NN, p_ER)
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

			print('Check Stochasticity:\n row: {} \n column {}'.format(np.sum(WW,axis=1),np.sum(WW,axis=0)))
			test = np.linalg.matrix_power((I_NN+Adj),NN)
			
			if np.all(test>0):
				print("the graph is connected\n")
				break 
			else:
				print("the graph is NOT connected\n")
				# quit()
		return G, Adj, E, Deg, WW

	def create_agents(self,NN):
		dim = self.agent_dim
		agents = []
		self.Q = np.zeros((NN,dim,dim))#10*np.random.rand(NN,dim, dim) ########CENTRALIZED###########
		self.R = 10*(np.random.rand(NN,dim)-1)
		for i in range(NN):
			T = scipy.linalg.orth(np.random.rand(dim,dim))
			D = np.diag(np.random.rand(dim))*10
			self.Q[i] = T.T@D@T#0.5*(self.Q[i].T+self.Q[i])
			agents.append(Agent(self.Q[i].copy(), self.R[i].copy().reshape((-1,1)), dim))
			self.XX[:,i,0] = agents[-1].get_state().flatten()
			self.YY[:,i,0] = agents[-1].get_gradient().flatten()
		return agents

	def update(self,iter):
		if iter == self.max_iters-1:
			for ii in range(NN):
				Nii = np.nonzero(self.Adj[ii])[0]
				_, yy, f_ii = self.agents[ii].update(ii,self.WW,self.lr,Nii,self.agents)
				self.FF[-1] += f_ii
				self.FF_grad[-1] += np.linalg.norm(yy)
			return

		for ii in range(NN):
			Nii = np.nonzero(self.Adj[ii])[0]
			y = self.agents[ii].update(ii,self.WW,self.lr,Nii,self.agents)
			self.XX[:,ii,iter+1], self.YY[:,ii,iter+1], f_ii = y[0].flatten(), y[1].flatten(), y[2]
			self.FF[iter] += f_ii
			self.FF_grad[iter] += np.linalg.norm(self.YY[:,ii,iter+1])
	def get_opt_cost(self):
		Q, R = self.Q.sum(axis = 0), self.R.sum(axis = 0) 
		xopt = -np.linalg.inv(Q)@R 
		fopt = 0.5*(xopt@Q@xopt) + R@xopt
		return fopt

	def plotCostEvolution(self):
		fopt = self.get_opt_cost()
		plt.figure()
		plt.semilogy(np.arange(self.max_iters), np.abs(self.FF-fopt))
		plt.xlabel(r"iterations $t$")
		plt.ylabel(r"$x_{i,t}$")
		plt.title("Evolution of the cost")
		plt.grid()
		plt.savefig('Figures/Cost_Evolution_task_1_1.png')

		plt.figure()
		plt.semilogy(np.arange(self.max_iters), self.FF_grad)
		plt.xlabel(r"iterations $t$")
		plt.ylabel(r"$loss_{i,t}$")
		plt.title("Evolution of the gradient of the cost")
		plt.grid()
		plt.savefig('Figures/Gradient_of_the_Cost_Evolution_task_1_1.png')
		

		plt.figure()
		plt.plot(np.arange(self.max_iters), np.repeat(fopt,self.max_iters), '--', linewidth=3)
		plt.plot(np.arange(self.max_iters), self.FF)
		plt.show()


if __name__ == '__main__':
	# Useful constants
	MAXITERS = np.int(1e4) # Explicit Casting
	NN = 200
	agent_dim = 5
	GT = GradientTracking(agent_dim, NN, MAXITERS, lr = 1e-4)
	for iter in tqdm(range(MAXITERS)):
		if (iter % 50) == 0:
			tqdm.write("Iteration {:3d}".format(iter), end="\n")
		GT.update(iter)
	GT.plotCostEvolution()