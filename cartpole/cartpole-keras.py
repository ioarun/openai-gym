import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import random
import gym

env = gym.make("CartPole-v0")
#Neural Nets as Q - function
model = Sequential()
model.add(Dense(500, init='lecun_uniform', input_shape=(4,)))
model.add(Activation('relu'))

model.add(Dense(500, init='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(2, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

epochs = 5000 # initial values : 1000
gamma = 0.9
epsilon = 1
j = 0
training_batch_x = []
training_batch_y = []
for i in range(epochs):
	state = np.array(env.reset())
	done = False
	#while game still in progress
	while(done == False):
		env.render()
		#we are in state S
		#let's run our Q function on S to get Q values for all possible
		#actions
		qval = model.predict(state.reshape(1,4), batch_size=1)
		if(random.random() < epsilon): #choose random action
			action = env.action_space.sample()
			print ("RANDOM ACTION :",action)
		else: #choose best action from Q(s,a) values
			action = (np.argmax(qval))

		#Take action, observe new state S'
		new_state, reward, done, info = env.step(action)
		#get max_Q(S',a)
		newQ = model.predict(new_state.reshape(1,4),batch_size=1)
		maxQ = np.max(newQ)
		y = np.zeros((1,2))
		y[:] = qval[:]
		if done == False: #non-terminal state
			update = (reward + (gamma*maxQ))
		else: #terminal state
			update = reward
		y[0][action] = update # target output
		#print ("Game #: %s" %(i,))
		training_batch_x.append(state.reshape(1,4))
		training_batch_y.append(y)
		model.fit(state.reshape(1,4), y, batch_size=1, nb_epoch=1, verbose=1)
		state = new_state
		#if reward != 1.0:
			#done = True
	if epsilon > 0.1:
		epsilon -= (1/epochs)


def testAlgo(init=0):
	done = False
	state = np.array(env.reset())
	#while the game still in progress
	while(done == False):
		env.render()
		qval = model.predict(state.reshape(1,4), batch_size=1)
		action = (np.argmax(qval))
		new_state, reward, done, info = env.step(action)
		state = np.array(new_state)
		#if reward != 1.0:
			#done = True
			#print("Reward: %s" %(reward,))

print ("TRAINING ENDED!")
testAlgo(init=0)