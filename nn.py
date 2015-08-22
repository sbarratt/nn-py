import numpy as np
from scipy.special import expit
import IPython as ipy
import pickle

class TwoLayerNeuralNetwork:
	def __init__(self, n_in, n_hid, n_out, eta, epochs, bin_size=None):
		self.n_in = n_in
		self.n_hid = n_hid
		self.n_out = n_out
		self.w_ih = .001*np.random.randn(self.n_in+1,self.n_hid)
		self.w_ho = .001*np.random.randn(self.n_hid+1,self.n_out)
		self.eta = eta
		self.epochs = epochs
		self.bin_size = bin_size

	def feedforward(self, x):
		x = np.r_[np.array([1]),x]
		h = np.tanh(self.w_ih.T.dot(x))
		h = np.r_[np.array([1]),h]
		o = expit(self.w_ho.T.dot(h))
		return x,h,o

	def backpropogate(self, x, h, o, y):
		Eo = (o-y)*o*(1-o)
		Eo, h = Eo.reshape(Eo.shape[0],1), h.reshape(h.shape[0],1)
		dwho = h.dot(Eo.T)

		Eh = (1-h[1:]**2) * self.w_ho[1:,:].dot(Eo)
		x, Eh = x.reshape(x.shape[0],1), Eh.reshape(Eh.shape[0],1)
		dwih = x.dot(Eh.T)

		return dwih, dwho

	def calc_error(self, data, labels):
		e = 0.0
		for i in range(data.shape[0]):
			y = np.zeros(self.n_out)
			y[labels[i]] = 1
			_,_, o = self.feedforward(data[i,:])
			e += (0.5)*np.linalg.norm(o-y)**2
		return e

	def train(self, data, labels, test_data, test_labels, save_weights=False):
		datapoints = data.shape[0]
		for j in range(self.epochs):
			for i in range(datapoints):
				x, h, o = self.feedforward(data[i,:])
				y = np.zeros(self.n_out)
				y[labels[i]] = 1
				dwih, dwho = self.backpropogate(x,h,o,y)
				ipy.embed()
				self.w_ih -= self.eta*dwih
				self.w_ho -= self.eta*dwho
			if save_weights:
				pickle.dump(self.w_ih,open("w_ih.pickle","wb"))
				pickle.dump(self.w_ho,open("w_ho.pickle","wb"))
			print np.sum(self.predict(test_data).ravel() == test_labels), "/", test_data.shape[0]

	def predict(self, data):
		pred = []
		for i in range(data.shape[0]):
			_,_,o = self.feedforward(data[i,:])
			pred.append(np.argmax(o))
		return np.array(pred)

class TwoLayerNeuralNetworkSoftMax:
	def __init__(self, n_in, n_hid, n_out, eta, epochs, bin_size=None):
		self.n_in = n_in
		self.n_hid = n_hid
		self.n_out = n_out
		self.w_ih = .001*np.random.randn(self.n_in+1,self.n_hid)
		self.w_ho = .001*np.random.randn(self.n_hid+1,self.n_out)
		self.eta = eta
		self.epochs = epochs
		self.bin_size = bin_size

	def feedforward(self, x):
		x = np.r_[np.array([1]),x]
		h = np.tanh(self.w_ih.T.dot(x))
		h = np.r_[np.array([1]),h]
		o_prime = self.w_ho.T.dot(h)
		o = np.exp(o_prime)*1.0/np.sum(np.exp(o_prime))
		return x,h,o

	def backpropogate(self, x, h, o, y):
		Eo = (o-y)
		Eo, h = Eo.reshape(Eo.shape[0],1), h.reshape(h.shape[0],1)
		dwho = h.dot(Eo.T)

		Eh = (1-h[1:]**2) * self.w_ho[1:,:].dot(Eo)
		x, Eh = x.reshape(x.shape[0],1), Eh.reshape(Eh.shape[0],1)
		dwih = x.dot(Eh.T)

		return dwih, dwho

	def calc_error(self, data, labels):
		e = 0.0
		for i in range(data.shape[0]):
			y = np.zeros(self.n_out)
			y[labels[i]] = 1
			_,_, o = self.feedforward(data[i,:])
			e -= np.log(o).dot(y)
		return e

	def train(self, data, labels, test_data, test_labels, save_weights=False):
		datapoints = data.shape[0]
		for j in range(self.epochs):
			for i in range(datapoints):
				x, h, o = self.feedforward(data[i,:])
				y = np.zeros(self.n_out)
				y[labels[i]] = 1
				dwih, dwho = self.backpropogate(x,h,o,y)
				self.w_ih -= self.eta*dwih
				self.w_ho -= self.eta*dwho
			if save_weights:
				pickle.dump(self.w_ih,open("w_ih.pickle","wb"))
				pickle.dump(self.w_ho,open("w_ho.pickle","wb"))
			print np.sum(self.predict(test_data).ravel() == test_labels), "/", test_data.shape[0]

	def predict(self, data):
		pred = []
		for i in range(data.shape[0]):
			_,_,o = self.feedforward(data[i,:])
			pred.append(np.argmax(o))
		return np.array(pred)

"""
def numerical_grad(self, x, label):
	e = 1e-5
	y = np.zeros(self.n_out)
	y[label] = 1
	dwih = np.zeros(self.w_ih.shape)
	for i in range(self.w_ih.shape[0]):
		for j in range(self.w_ih.shape[1]):
			self.w_ih[i,j] += e
			f_plus = 0.5*np.linalg.norm(self.feedforward(x)[2]-y)**2
			self.w_ih[i,j] -= 2*e
			f_minus = 0.5*np.linalg.norm(self.feedforward(x)[2]-y)**2
			self.w_ih[i,j] += e
			dwih[i,j] = (f_plus-f_minus)*1.0/(2*e)
	dwho = np.zeros(self.w_ho.shape)
	for i in range(self.w_ho.shape[0]):
		for j in range(self.w_ho.shape[1]):
			self.w_ho[i,j] += e
			f_plus = 0.5*np.linalg.norm(self.feedforward(x)[2]-y)**2
			self.w_ho[i,j] -= 2*e
			f_minus = 0.5*np.linalg.norm(self.feedforward(x)[2]-y)**2
			self.w_ho[i,j] += e
			dwho[i,j] = (f_plus-f_minus)*1.0/(2*e)
	return dwih, dwho

def train_batch(self, data, labels, test_data, test_labels):
	for _ in range(self.epochs):
		for j in range(data.shape[0]/self.bin_size):
			dwih, dwho = np.zeros(self.w_ih.shape),np.zeros(self.w_ho.shape)
			for i in range(j*self.bin_size,(j+1)*self.bin_size):
				x, h, o = self.feedforward(data[i,:])
				y = np.zeros(self.n_out)
				y[labels[i]] = 1
				dwih1, dwho1 = self.backpropogate(x,h,o,y)
				dwih, dwho = dwih + dwih1, dwho + dwho1
			self.w_ih -= self.eta*dwih*1.0/self.bin_size
			self.w_ho -= self.eta*dwho*1.0/self.bin_size
		print np.sum(self.predict(test_data).ravel() == test_labels), "/", test_data.shape[0]
"""

