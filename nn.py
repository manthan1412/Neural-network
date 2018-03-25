import numpy as np

class NeuralNetwork(object):
	
	def __init__(self):
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

	def __train(self, data, target, verbose):
		if verbose:
			print("Training a model")
		for _ in range(self.__epoch):
			for i in range(self.__N):
				# feed_forward(data[i])
				# calculate_error()
				# backpropagate()
				pass
		pass

	def fit(self, data, target, eta = 0.1, epoch=1000, verbose = False):
		
		self.__N = len(data)
		self.__dim = len(data[0])
		self.__eta = eta
		self.__epoch = epoch

		# init model here
		self.__model = self.__init_model()

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