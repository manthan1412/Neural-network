import numpy as np
from numpy.random import uniform as random 

class NeuralNetwork(object):
	
	def __init__(self):
		self.__W = []
		self.__low = -0.01
		self.__high = 0.01
		self.__bias = []
		self.__s = []
		self.__x = []
		pass

	def __init_model(self):
		model = []

		# input layer
		# w length dimension

		# hidden layer
		# w length 2

		# output layer
		# w length 100 + 1(bias)
		return model

	def __feed_forward(self, data):
		n = len(self.__W)

		# input layer
		i = 0
		for perceptron in self.__W[0]:
			for weight in perceptron:
				self.__s[0][i] = weight*data[i]

				## convert s to x
				i += 1
		print(self.__s)
		print(i)

		for i in range(1, n):
			for perceptron in self.__W[i]:
				for weight in perceptron:
					# print(weight)
					# if i == 0:

					pass
		pass

	def __back_propagation(self, data):
		pass

	def __train(self, data, target, verbose):
		if verbose:
			print("Training a model")
		for _ in range(self.__epoch):
			for i in range(self.__N):
				self.__feed_forward(data[i])
				# calculate_error()
				self.__back_propagation(data[i])
				break
				pass
			break
		pass

	def add_layer(self, size, input_size = None, type='hidden'):
		# print(input_size)
		if not input_size:
			input_size = len(self.__W[-1])
		# print(input_size)
		weights = []
		if type == 'input':
			
			for _ in range(input_size):
				weight = random(self.__low, self.__high, size)
				weights.append(weight)
			self.__bias.append([1])
			s = random(0, 0, input_size)
			x = random(0, 0, input_size)
			self.__s.append(s)
			self.__x.append(x)
			# weights[0] = weights[0].tolist()
			# weights[0].insert(0, 1)
			# weights[0] = np.array(weights[0])
			# print(weights[0])
		
		if type == 'hidden':
			for _ in range(size):
				weight = random(self.__low, self.__high, input_size)
				weights.append(weight)
			self.__bias.append([])
			s = random(0, 0, size)
			x = random(0, 0, size)
			self.__s.append(s)
			self.__x.append(x)

		if type == 'output':
			for _ in range(size):
				weight = random(self.__low, self.__high, input_size)
				weights.append(weight)
			self.__bias.append([1])
			s = random(0, 0, size)
			x = random(0, 0, size)
			self.__s.append(s)
			self.__x.append(x)

		self.__W.append(weights)


	def fit(self, data, target, eta = 0.1, epoch=1000, verbose = False):
		
		self.__N = len(data)
		self.__dim = len(data[0])
		self.__eta = eta
		self.__epoch = epoch
		self.__verbose = verbose

		# init model here
		# self.__model = self.__init_model()
		print(len(self.__W[2][0]))
		print(self.__bias)
		print(self.__s[1][0], len(self.__x[0]))
		self.__train(data, target, verbose)

	def predict(self, data):
		
		labels = []
		n = len(data)
		# use self.__W
		for i in range(n):
			# labels.append(predict_label(data[i]))
			pass
		return np.array(labels)

	def accuracy(self, true_target, predicted_target):
		accuracy = 1
		pass
		return accuracy
