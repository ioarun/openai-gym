'''
Breakout-v0 using Full Deep Q Learning
observation dimensions (210, 160, 3)
actions ['NOOP', 'FIRE','RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

'''
import tensorflow as tf
import gym
import numpy as np
import math
import random
from matplotlib import pyplot as plt

env = gym.make("Breakout-v0")

#observation = env.reset()

#print env.get_action_meanings()

#plt.figure()
#plt.imshow(env.render(mode='rgb_array'))

#[env.step(4) for x in range(1)]
#plt.figure()
#plt.imshow(env.render(mode='rgb_array'))
#plt.imshow(observation[34:-16,:,:])
#plt.imshow(observation)
#env.render(close=True)

#plt.show()

VALID_ACTIONS = [0, 1, 2, 3]

# input_preprocessor graph
input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
output = tf.image.rgb_to_grayscale(input_state)
# image, offset_height, offset_width, target_height, target_width
output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
output = tf.image.resize_images(output, [84, 84])
output = tf.squeeze(output)

# build model
# input is 4 grayscale frames of 84, 84 each
X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name='X')
# target value
y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name='y')
# which action was chosen
actions_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="actions")

X = tf.to_float(X_pl) / 255.0
batch_size = tf.shape(X_pl)[0]

# three convolutional layers
conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

# fully connected layers
flattened = tf.contrib.layers.flatten(conv3)
fc1 = tf.contrib.layers.fully_connected(flattened, 512)
predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

# get predictions for chosen actions only
gather_indices = tf.range(batch_size)*tf.shape(predictions[1] + actions_pl)
action_predictions = tf.gather(tf.reshape(predictions, [-1]), gather_indices)

# calculate loss
losses = tf.squared_difference(y_pl, action_predictions)
loss = tf.reduce_mean(losses)

# optimizer parameters
optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
train_op = optimizer.minimize(loss, global_step=tf.contrib.framework.get_global_step())


def input_preprocessor(sess, state):
	return sess.run(output, feed_dict={input_state: state})
	
def predict(sess, s):
	'''
	s: shape [batch_size, 4, 160, 160, 3]
	returns: shape[batch_size, NUM_VALID_ACTIONS]
	'''
	return sess.run(predictions, feed_dict={X_pl: s})

def update(sess, s, a, y):
	'''
	s: shape [batch_size, 4, 160, 160, 3]
	a: chosen actions of shape [batch_size]
	y: targets of shape [batch_size]
	returns: calculated loss on the batch
	'''
	_, _loss = sess.run([train_op, loss], feed_dict={X_pl: s, y_pl: y, actions_pl: a})
	return _loss

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	observation = env.reset()
	observation_p = input_preprocessor(sess, observation)
	observation = np.stack([observation_p]*4, axis=2)
	observations = np.array([observation]*2)

	# test predictions
	print (predict(sess, observations))

	# test training step
	y = np.array([10.0, 10.0])
	a = np.array([1, 3])
	print (update(sess, observations, a, y))
	
	
	