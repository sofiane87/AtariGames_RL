
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from time import time
import platform
currentSystem = platform.system()

################### BUILDING PATH ############################

plotFolderPath = 'Q6' +  os.path.sep + 'Plot' + os.path.sep
numpyFolderPath = 'Q6' +  os.path.sep +  'Numpy' + os.path.sep
modelFolderPath = 'Q6' +  os.path.sep +  'Model' + os.path.sep


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
learningRate = 0.015
nEpisodes = 2000
nIterations = 1
std = learningRate/10
useBias = False
useStop = True
useBatchNorm = False
epsilon = 0.05
hiddenSize = 100

global firstRun 
global bufferSize 
global transitionIndex
global bufferObservations 
global bufferNextObservations 
global bufferActions 
global bufferRewards 
global BatchSize 

bufferSize = 2000
firstRun = True
transitionIndex = 0
bufferObservations = np.zeros([bufferSize, 4])
bufferNextObservations = np.zeros([bufferSize, 4])
bufferActions = np.zeros([bufferSize])
bufferRewards = np.zeros([bufferSize])
BatchSize = 500


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
			if (done or (t + 1 == maximumLength)):
				if t+1 != maximumLength:
					rewardData.append(-1)
					nextObservationData.append([0 for obs in observation])

				else:
					rewardData.append(0)
					nextObservationData.append([obs for obs in observation])

				#print("Episode "+ str(i_episode)+" finished after {} timesteps".format(t+1))
				#print("Reward : "+ str(returnReward[i_episode]))
				break
			else:
				rewardData.append(0)
				nextObservationData.append([obs for obs in observation])


	print('DONE')
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
			if (done or (t + 1 == maximumLength)):
				episodeLength[i_episode] = t +1 
				if (t + 1 != maximumLength):
					rewards[i_episode] = -(discountFactor)**(t)
				break 

	return rewards, episodeLength


def trainAnEpisode(sess, loss_var, bellman_loss_var, train_op, qVector, episode):

	global firstRun
	global transitionIndex
	global bufferObservations
	global bufferNextObservations
	global bufferActions
	global bufferRewards
	global BatchSize
	global bufferSize
	#print('EPISODE : ', i_episode)
	observation = env.reset()
	
	loss = []
	bellmanLoss = []

	for t in range(maximumLength):

		# env.render()
		randValue = np.random.random(1)
		if randValue <= epsilon:
			action = np.random.random_integers(0,1)
		else:
			action = greedy_action(observation, sess, qVector)

		new_observation, reward, done, info = env.step(action)
		episodeLength = t +1 

		if (done and (t + 1 != maximumLength)):
			new_observation = [0  for obs in new_observation]
			tempreward = -1
			reward = -(discountFactor)**(t)
		else : 
			tempreward = 0
			reward = 0


		if firstRun:
			bufferObservations[:] = observation
			bufferNextObservations[:] = new_observation
			bufferActions[:] = action
			bufferRewards[:] = tempreward
			firstRun = False
		else:
			bufferObservations[transitionIndex] = observation
			bufferNextObservations[transitionIndex] = new_observation
			bufferActions[transitionIndex] = action
			bufferRewards[transitionIndex] = tempreward

		############ BATCHING #########################
		batchIndexes = np.random.choice(range(bufferSize), size=BatchSize, replace=False)
		observationBatch = bufferObservations[batchIndexes]
		nextobservationBatch = bufferNextObservations[batchIndexes]
		actionBatch = bufferActions[batchIndexes]
		rewardBatch = bufferRewards[batchIndexes]
		indiceBatch = np.array(range(BatchSize))
		transitionIndex += 1
		transitionIndex = transitionIndex % bufferSize

		feed_dict = {currentState : observationBatch, 
					 is_training_bool : True, 
					 currentAction : actionBatch, 
					 rowIndices : indiceBatch, 
					 nextState : nextobservationBatch, 
					 reward_value : rewardBatch
					 }
		observation = new_observation
		temploss, tempBellmanLoss, _ = sess.run([loss_var,bellman_loss_var, train_op], feed_dict=feed_dict)
		loss.append(temploss) 
		bellmanLoss.append(tempBellmanLoss) 

		if (done or (t + 1 == maximumLength)):
			break;

	loss = np.mean(loss)
	bellmanLoss = np.mean(bellmanLoss)
	if episode % 20 == 0:
		_ , testEpisodeLength = greedy_policy(10, 300, sess, qVector)
		testEpisodeLength =  np.mean(testEpisodeLength)
		return loss, bellmanLoss, reward, episodeLength, testEpisodeLength
	else:
		return loss, bellmanLoss, reward, episodeLength


def savePlots(inputArray, filename, nEpochs, ylabel, mode = 'step'):
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

	# Output Hidden Layer
	qHiddenWeight = tf.Variable(tf.random_normal([4, hiddenSize], stddev=std))
	if useBias:
		qHiddenBias = tf.Variable(tf.zeros([hiddenSize]))
		qHiddenMatrix = tf.nn.relu(tf.matmul(currentState, qHiddenWeight) + qHiddenBias)
	else:
		qHiddenMatrix = tf.nn.relu(tf.matmul(currentState, qHiddenWeight))


	if useBatchNorm:
		qHiddenMatrix = batch_norm_wrapper(qHiddenMatrix, is_training_bool)

	# Output Q matrix for all actions 


	qWeight = tf.Variable(tf.random_normal([hiddenSize, 2], stddev=std))
	if useBias:
		qBias = tf.Variable(tf.zeros([2]))
		qMatrix = tf.matmul(qHiddenMatrix, qWeight) + qBias
	else:
		qMatrix = tf.matmul(qHiddenMatrix, qWeight)


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
		nextqHiddenMatrix = tf.nn.relu(tf.matmul(nextState, qHiddenWeight) + qHiddenBias)
	else:
		nextqHiddenMatrix = tf.nn.relu(tf.matmul(nextState, qHiddenWeight))

	if useBatchNorm:
		nextqHiddenMatrix = batch_norm_wrapper(nextqHiddenMatrix, is_training_bool)

	if useBias:
		nextQ = tf.matmul(nextqHiddenMatrix, qWeight) + qBias
	else:
		nextQ = tf.matmul(nextqHiddenMatrix, qWeight)


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
	train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)


	return currentState, nextState, is_training_bool, reward_value, currentAction, rowIndices, loss, bellman_loss, train_step, qMatrix

#################################### GENERATING THE DATA ######################################


#################################### GENERATING THE DATA ######################################


bellmanLossArray = np.zeros([nIterations, nEpisodes])
lossArray = np.zeros([nIterations, nEpisodes])
stepArray = np.zeros([nIterations, nEpisodes])
testStepArray = np.zeros([nIterations, int(nEpisodes/20)])
rewardArray = np.zeros([nIterations, nEpisodes])

print('STARTING TRAINING')


print('LEARNING RATE : ', learningRate)

currentState, nextState, is_training_bool, reward_value, currentAction, rowIndices, loss, bellman_loss, train_step, qMatrix = buildNetwork()

for iteration in range(nIterations):
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	count = 0
	bestValue = 0
	saverPath = modelFolderPath + 'Model_{}_{}.ckpt'.format(iteration, learningRate)
	maxCount = 0

	with tf.Session() as sess :
		sess.run(init)
		for episode in range(nEpisodes):
			initialTime = time()

			if episode % 20 == 0:
				lossArray[iteration,episode], bellmanLossArray[iteration,episode], rewardArray[iteration,episode], stepArray[iteration,episode], testStepArray[iteration,int(episode/20)] = trainAnEpisode(sess, loss, bellman_loss,train_step, qMatrix,episode)
				saveText = '   Saved Results: {}           '.format(bestValue)
				if bestValue < testStepArray[iteration,int(episode/20)]:
						bestValue = testStepArray[iteration,int(episode/20)]
						saver.save(sess, saverPath)
						saveText =  '   Saved Results: {} - Saved'.format(bestValue)
						count = 1
				elif bestValue == testStepArray[iteration,int(episode/20)]:
					count += 1
					maxCount = max(count,maxCount)
					if count >= maxCount:
						saver.save(sess, saverPath)
						saveText =  '   Saved Results: {} - Saved'.format(bestValue)
					else:
						saveText = '   Saved Results: {} - Stopped'.format(bestValue)
				elif testStepArray[iteration,int(episode/20)] <= bestValue - 50:
						count = 0
				maxCount = max(count,maxCount)

			else:
				lossArray[iteration,episode], bellmanLossArray[iteration,episode], rewardArray[iteration,episode], stepArray[iteration,episode] = trainAnEpisode(sess, loss, bellman_loss, train_step, qMatrix,episode)
			
			
			if currentSystem == 'Windows':
				os.system('cls')
			else:
				os.system('clear')

			timePerEpisode = time() - initialTime
			print('Episode : {}    Reward : {:.5f}    Loss : {:.5f}     Bellman-Loss : {:.5f}  \nTrain-Steps : {}   Test-Steps : {}  Time per episode : {:.1f}s  Longest Streak : {}'.format(episode, rewardArray[iteration,episode],lossArray[iteration,episode],bellmanLossArray[iteration,episode],stepArray[iteration,episode],testStepArray[iteration,int(np.floor(episode/20))],timePerEpisode, maxCount) + saveText)



	np.save(numpyFolderPath + 'bellmanLossArray_{}'.format(learningRate), bellmanLossArray)
	np.save(numpyFolderPath + 'lossArray_{}'.format(learningRate), lossArray)
	np.save(numpyFolderPath + 'rewardArray_{}'.format(learningRate), rewardArray)
	np.save(numpyFolderPath + 'train_stepArray_{}'.format(learningRate), stepArray)
	np.save(numpyFolderPath + 'test_stepArray_{}'.format(learningRate), testStepArray)


savePlots(np.mean(bellmanLossArray, axis = 0), plotFolderPath + 'bellmanLossPlot_{}'.format(learningRate), nEpisodes,'bellman loss', mode = '')
savePlots(np.mean(lossArray, axis = 0), plotFolderPath + 'lossPlot_{}'.format(learningRate),nEpisodes, 'loss', mode = '')
savePlots(np.mean(rewardArray, axis = 0), plotFolderPath + 'rewardPlot_{}'.format(learningRate),nEpisodes, 'cumulative reward', mode = 'reward')
savePlots(np.mean(stepArray, axis = 0), plotFolderPath + 'train_stepPlot_{}'.format(learningRate),nEpisodes, 'train steps before termination', mode = 'step')
savePlots(np.mean(testStepArray, axis = 0), plotFolderPath + 'test_stepPlot_{}'.format(learningRate),int(nEpisodes/20), 'test steps before termination', mode = 'step')


