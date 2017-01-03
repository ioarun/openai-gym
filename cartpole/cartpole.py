'''
The state features are : [cart_position, pole_angle, cart_velocity, angle_rate_of_change]
Find the OpenAI gym at https://gym.openai.com/evaluations/eval_IkBgtzYaRNahmH9hra0gw
'''

from neuralnet import Network
import gym
import numpy as np

env = gym.make("CartPole-v0")

alpha = 0.85
gamma = 0.99
num_episodes = 5000
epsilon = 1.0
eta = 1.0
net = Network([4, 20,20, 2])

for _ in range(num_episodes):
	_ += 1
	state = env.reset()
	i = 0
	r = 0
	while(i < 99):
		env.render()
		qval = np.array(net.feedforward((np.array(state)).reshape(4,1)))
		if np.random.random() > epsilon:
			action = np.argmax(qval)
		else:
			action = env.action_space.sample()
		#print "taking action :",action
		observation, reward, done, info = env.step(action)
		state = observation
		newQval = np.array(net.feedforward((np.array(observation)).reshape(4,1)))
		update = reward + gamma*(np.max(newQval))
		y = np.zeros((2,1))
		y[:] = qval[:]
		y[action] = update
		training_data = [[np.array(state).reshape(4,1),y]]
		#print training_data
		net.gradient_descent(training_data, eta)
		r += reward
		if done == True:
			break
	print "Episode ",_," completed."

	if epsilon >= 0.1:
		epsilon -= 1.0/num_episodes
print "TRAINING ENDED"
print net.weights
print net.biases
