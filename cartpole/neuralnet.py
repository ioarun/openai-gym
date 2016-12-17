import numpy as np

class Network(object):
		def __init__(self, sizes):
			self.sizes = sizes
			self.num_layers = len(sizes)
			self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
			self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

		def feedforward(self, a):
			for b, w in zip(self.biases, self.weights):
				a = sigmoid(np.dot(w, a) + b)
			return a

		def gradient_descent(self, training_data, eta):
			self.update_weights(training_data, eta)


		def update_weights(self, training_data, eta):
			nabla_w = [np.zeros(w.shape) for w in self.weights]
			nabla_b = [np.zeros(b.shape) for b in self.biases]

			delta_nabla_w, delta_nabla_b = self.backprop(training_data)
			nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
			nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]

			self.weigths = [w - (eta*nw) for w,nw in zip(self.weights, nabla_w)]
			self.biases = [b - (eta*nb) for b,nb in zip(self.biases, nabla_b)]

		def backprop(self, training_data):
			nabla_w = [np.zeros(w.shape) for w in self.weights]
			nabla_b = [np.zeros(b.shape) for b in self.biases]
			x = training_data[0][0]
			y = training_data[0][1]

			activation = x
			activations = [x]
			zs = []

			for b, w in zip(self.biases, self.weights):
				z = np.dot(w,activation) + b
				zs.append(z)
				activation = sigmoid(z)
				activations.append(activation)
			#print "activations :",activations
			delta = (activations[-1] - y)*sigmoid_prime(zs[-1])

			nabla_b[-1] = delta
			nabla_w[-1] = np.dot(delta, activations[-2].transpose())

			for l in xrange(2, self.num_layers):
				sp = sigmoid_prime(zs[-l])
				delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
				nabla_b[-l] = delta
				nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

			return (nabla_w, nabla_b)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))
