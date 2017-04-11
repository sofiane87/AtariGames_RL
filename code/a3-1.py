
from gym import envs
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import platform

from time import time
currentSystem = platform.system()
################### BUILDING PATH ############################


plotFolderPath = 'Q3' +  os.path.sep  + 'Part1' + os.path.sep + 'Plot' + os.path.sep
numpyFolderPath = 'Q3' +  os.path.sep  + 'Part1' +   os.path.sep +  'Numpy' + os.path.sep
modelFolderPath = 'Q3' +  os.path.sep  + 'Part1' +   os.path.sep +  'Model' + os.path.sep
if not(os.path.exists(plotFolderPath)):
	os.makedirs(plotFolderPath)

if not(os.path.exists(numpyFolderPath)):
	os.makedirs(numpyFolderPath)

if not(os.path.exists(modelFolderPath)):
	os.makedirs(modelFolderPath)

################# PARAMETERS ##########################
argList = sys.argv[1:]

discountFactor = 0.99
numberOfEpisodes = 2000
maximumLength = 300
learningRates = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 0.5]
lr_index = 0

if ('-i' in argList):
	lr_index = int(argList[argList.index('-i') + 1])

learningRate = learningRates[min(lr_index,len(learningRates)-1)]
nEpochs = 100
batchSize = 500
std = learningRate
nonBatch = False
useBias = False
useStop = True
useBatchNorm = False
optimizer = 'gradient'

saverPath = modelFolderPath + 'Model_{}.ckpt'.format(learningRate)

################### Initializing GYM ##########################
gym.envs.register(
	id = 'CartPoleModified-v0',
	entry_point = 'gym.envs.classic_control:CartPoleEnv',
	max_episode_steps = maximumLength,
)
env = gym.make('CartPoleModified-v0')

#################### Defining functions #####################

def train_batch_norm_wrapper(inputs, scale, beta, pop_mean, pop_var, decay=0.95, epsilon=1e-3):
	batch_mean, batch_var = tf.nn.moments(inputs, [0])
	train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
	train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
	with tf.control_dependencies([train_mean, train_var]):
		return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)



def batch_norm_wrapper(inputs, is_training, decay=0.95, epsilon=1e-3):
	scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
	return tf.cond(is_training, lambda : train_batch_norm_wrapper(inputs, scale, beta, pop_mean, pop_var), lambda: tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon))


def random_policy(numberOfEpisodes, maximumLength):

	############### INITIALIZING ##################
	actionData = []
	rewardData = []
	observationData = []
	nextObservationData = []
	infoData = []

	################ SIMULATING ##################


	for i_episode in range(numberOfEpisodes):
		#print('EPISODE : ', i_episode)
		observation = env.reset()
		for t in range(maximumLength):
			# env.render()
			action = np.random.random_integers(0,1)
			actionData.append(action)
			observationData.append([obs for obs in observation])
			observation, reward, done, info = env.step(action)
			infoData.append(infoData)
			if (done and (t + 1 != maximumLength)):
				rewardData.append(-1)
				nextObservationData.append([0 for obs in observation])

				#print("Episode "+ str(i_episode)+" finished after {} timesteps".format(t+1))
				#print("Reward : "+ str(returnReward[i_episode]))
				break
			elif (t+1 == maximumLength):
				print(observation)
				nextObservationData.append([obs for obs in observation])
				rewardData.append(0)
				break
			else:
				nextObservationData.append([obs for obs in observation])
				rewardData.append(0)

	print('number Of Actions: ', len(actionData))

	############### CHAGING INTO ARRAY ##################
	# print('ACTION ARRAY')
	actionData = np.array(actionData)
	# print('REWARD ARRAY')
	rewardData = np.array(rewardData)
	# print('OBSERVATION ARRAY')
	observationData = np.array(observationData)
	# print('NEXT OBSERVATION ARRAY')
	nextObservationData = np.array(nextObservationData)
	# print('ARRAY BUILDING DONE')
	#infoData = np.array(infoData)
	return actionData, rewardData, observationData, nextObservationData

def greedy_action(observation, sess, qVector):
	return np.argmax(sess.run(qVector, feed_dict = {currentState : np.array([observation]), is_training_bool : False}))


def greedy_policy(numberOfEpisodes, maximumLength, sess, qVector):

	############### INITIALIZING ################## 
	episodeLength = np.zeros([numberOfEpisodes]).astype(int)
	rewards = np.zeros([numberOfEpisodes])
	################ SIMULATING ##################


	for i_episode in range(numberOfEpisodes):
		#print('EPISODE : ', i_episode)
		observation = env.reset()
		for t in range(maximumLength):
			# env.render()
			action = greedy_action(observation, sess, qVector)
			observation, reward, done, info = env.step(action)
			episodeLength[i_episode] = t +1 
			if (done and (t + 1 != maximumLength)):
				rewards[i_episode] = -(discountFactor)**(t)
				break 


	return rewards, episodeLength

def savePlots(inputArray, filename, ylabel, mode = 'step'):
	plot_array = inputArray
	if mode == 'step':
		plt.axis([0, nEpochs, 0, maximumLength])
	elif mode == 'reward':
		plt.axis([0, nEpochs, -1, 0])
	
	plt.plot(plot_array)
	plt.xlabel('number of epochs')
	plt.ylabel(ylabel)
	plt.savefig(filename + '.png')
	plt.close()

def saveTex (inputArray, filename, ylabel):
	print('build It')
	plot_array = inputArray
	plt.plot(plot_array)
	plt.xlabel('number of epochs')
	plt.ylabel(ylabel)
	plt.savefig(filename + '.png')
	plt.close()


#################### Building the Model ##################################

def buildNetwork():


	currentState = tf.placeholder(tf.float32, [None, 4])

	# Output Q matrix for all actions 
	qWeight = tf.Variable(tf.random_normal([4, 2], stddev=std))
	if useBias:
		qBias = tf.Variable(tf.zeros([2]))
		qMatrix = tf.matmul(currentState, qWeight) + qBias
	else:
		qMatrix = tf.matmul(currentState, qWeight)

	is_training_bool = tf.placeholder(tf.bool)

	if useBatchNorm:
		qMatrix = batch_norm_wrapper(qMatrix, is_training_bool)

	# Picking the proper QValues 

	currentAction = tf.placeholder(tf.int32, [None])
	rowIndices = tf.placeholder(tf.int32, [None])

	fullIndices = tf.stack([rowIndices, currentAction], axis = 1)
	predictedQValues = tf.gather_nd(qMatrix, fullIndices)


	# Computing the bestQvalue for the next state
	nextState = tf.placeholder(tf.float32, [None,4])
	if useBias:
		nextQ = tf.matmul(nextState, qWeight) + qBias
	else:
		nextQ = tf.matmul(nextState, qWeight)

	if useBatchNorm:
		nextQ = batch_norm_wrapper(nextQ, is_training_bool)

	if useStop:
		nextQ = tf.stop_gradient(nextQ)

	# Computing the target
	reward_value = tf.placeholder(tf.float32, [None])
	target_value = discountFactor*tf.reduce_max(nextQ, axis =1) + reward_value

	# Computing the loss
	loss = 0.5*tf.reduce_mean(tf.square(target_value - predictedQValues))
	bellman_loss = tf.reduce_mean(target_value - predictedQValues)
	# Train Step 
	if optimizer == 'adam':
		train_step = tf.train.AdamOptimizer(learningRate).minimize(loss)
	elif optimizer == 'rms':
		train_step = tf.train.RMSPropOptimizer(learningRate).minimize(loss)
	else:
		train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

	return currentState, nextState, is_training_bool, reward_value, currentAction, rowIndices, loss, bellman_loss, train_step, qMatrix

#################################### GENERATING THE DATA ######################################

actionData, rewardData, observationData, nextObservationData = random_policy(numberOfEpisodes, maximumLength)
currentState, nextState, is_training_bool, reward_value, currentAction, rowIndices, loss, bellman_loss, train_step, qMatrix = buildNetwork()

if nonBatch:
	batchSize = actionData.shape[0]


lossArray = np.zeros([nEpochs])
bellmanLossArray = np.zeros([nEpochs])
stepArray = np.zeros([nEpochs])
rewardArray = np.zeros([nEpochs])

print('STARTING TRAINING')


print('LEARNING RATE : ', learningRate)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
bestValue = 0
sess = tf.Session()
sess.run(init)
nBatches = int(np.ceil(actionData.shape[0]/batchSize))
for epoch in range(nEpochs):
	firstTime = time()
	trainLoss = 0
	trainBellmanLoss = 0
	randPerm = np.random.permutation(rewardData.shape[0])
	for batch in range(nBatches):
		initialIndex = batch * batchSize
		finalIndex = min((batch +1) * batchSize, actionData.shape[0])

		################# Extracting Minibatches #######################
		actionBatch = actionData[randPerm[initialIndex:finalIndex]]
		rewardBatch = rewardData[randPerm[initialIndex:finalIndex]]
		observationBatch = observationData[randPerm[initialIndex:finalIndex]]
		nextObservationBatch = nextObservationData[randPerm[initialIndex:finalIndex]]
		row_indices = np.array(range(finalIndex - initialIndex))
		####################### Training ############################
		tempBellmanLoss, temploss, _ = sess.run([bellman_loss,loss,train_step], feed_dict = {currentState : observationBatch, is_training_bool : False, currentAction : actionBatch, rowIndices : row_indices, nextState :nextObservationBatch, reward_value : rewardBatch})
		trainLoss += temploss/nBatches
		trainBellmanLoss += tempBellmanLoss/nBatches
	lossArray[epoch] = trainLoss
	bellmanLossArray[epoch] = trainBellmanLoss
	rewards, lengths = greedy_policy(20,maximumLength,sess,qMatrix)
	meanNumberOfSteps = np.mean(lengths)
	rewardArray[epoch] = np.mean(rewards)
	stepArray[epoch] = meanNumberOfSteps
	saved_string = '            '
	if bestValue < meanNumberOfSteps:
		bestValue = meanNumberOfSteps
		saver.save(sess, saverPath)
		saved_string = ' - Saved'
	
	if currentSystem == 'Windows':
		os.system('cls')
	else:
		os.system('clear')
	
	epochLength = time() - firstTime
	print('Epoch : {} Loss : {:.5f} bellman-loss : {:.5} \nSteps before termination : {:.3f}   Epoch time : {:.1f}s '.format(epoch,trainLoss,trainBellmanLoss,meanNumberOfSteps,epochLength)+saved_string)
	
savePlots(bellmanLossArray, plotFolderPath + 'bellmanLossPlot_{}'.format(learningRate), 'bellman loss', mode = '')
savePlots(lossArray, plotFolderPath + 'lossPlot_{}'.format(learningRate), 'loss', mode = '')
savePlots(rewardArray, plotFolderPath + 'rewardPlot_{}'.format(learningRate), 'cumulative reward', mode = 'reward')
savePlots(stepArray, plotFolderPath + 'stepPlot_{}'.format(learningRate), 'steps before termination', mode = 'step')

np.save(numpyFolderPath + 'bellmanLossArray_{}'.format(learningRate), bellmanLossArray)
np.save(numpyFolderPath + 'lossArray_{}'.format(learningRate), lossArray)
np.save(numpyFolderPath + 'rewardArray_{}'.format(learningRate), rewardArray)
np.save(numpyFolderPath + 'stepArray_{}'.format(learningRate), stepArray)

