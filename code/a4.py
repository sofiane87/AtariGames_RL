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


plotFolderPath = 'Q4' +  os.path.sep + 'Plot' + os.path.sep
numpyFolderPath = 'Q4' +  os.path.sep +  'Numpy' + os.path.sep
modelFolderPath = 'Q4' +  os.path.sep +  'Model' + os.path.sep


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
learningRate = 0.01
nEpisodes = 2000
nIterations = 100
firstIteration = 0
std = learningRate/10
useBias = False
useStop = True
useBatchNorm = False
epsilon = 0.05
hiddenSize = 100


if ('-iter' in argList):
	firstIteration = int(argList[argList.index('-iter')+1])


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
			elif(done and t+1 == maximumLength):
				rewardData.append(0)
				nextObservationData.append([obs for obs in observation])
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
				if t+1 != maximumLength:
					rewards[i_episode] = -(discountFactor)**(t)
				break 

	return rewards, episodeLength


def trainAnEpisode(sess, loss_var, bellman_loss_var , train_op, qVector):

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
			tempreward = -1
			reward = -(discountFactor)**(t)
			new_observation = [0 for obs in new_observation]
		else : 
			tempreward = 0
			reward = 0
		
		feed_dict = {currentState : np.array([observation]), 
					 is_training_bool : True, 
					 currentAction : np.array([action]), 
					 rowIndices : np.array([0]), 
					 nextState :np.array([new_observation]), 
					 reward_value : np.array([tempreward])
					 }
		observation = new_observation
		temploss, tempbellmanloss, _ = sess.run([loss_var, bellman_loss_var, train_op], feed_dict=feed_dict)
		loss.append(temploss) 
		bellmanLoss.append(tempbellmanloss)
		if (done or (t + 1 == maximumLength)):
			break;

	loss = np.mean(loss)
	bellmanLoss = np.mean(bellmanLoss)
	_ , testEpisodeLength = greedy_policy(10, 300, sess, qVector)
	testEpisodeLength =  np.mean(testEpisodeLength)
	return loss, bellmanLoss, reward, episodeLength, testEpisodeLength

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


lossArray = np.zeros([nIterations, nEpisodes])
bellmanLossArray = np.zeros([nIterations, nEpisodes])
stepArray = np.zeros([nIterations, nEpisodes])
testStepArray = np.zeros([nIterations, nEpisodes])
rewardArray = np.zeros([nIterations, nEpisodes])

if firstIteration != 0:
	bellmanLossArray = np.load(numpyFolderPath + 'bellmanLossArray_{}.npy'.format(learningRate))
	lossArray = np.load(numpyFolderPath + 'lossArray_{}.npy'.format(learningRate))
	rewardArray = np.load(numpyFolderPath + 'rewardArray_{}.npy'.format(learningRate))
	stepArray = np.load(numpyFolderPath + 'train_stepArray_{}.npy'.format(learningRate))
	testStepArray = np.load(numpyFolderPath + 'test_stepArray_{}.npy'.format(learningRate))


print('STARTING TRAINING')


print('LEARNING RATE : ', learningRate)

currentState, nextState, is_training_bool, reward_value, currentAction, rowIndices, loss, bellman_loss, train_step, qMatrix = buildNetwork()


for iteration in range(firstIteration,nIterations):
	

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
			lossArray[iteration,episode], bellmanLossArray[iteration, episode], rewardArray[iteration,episode], stepArray[iteration,episode], testStepArray[iteration,episode] = trainAnEpisode(sess, loss, bellman_loss, train_step, qMatrix)
			saveText = '   Saved Results: {}           '.format(bestValue)
			if bestValue < testStepArray[iteration,episode]:
					bestValue = testStepArray[iteration,episode]
					saver.save(sess, saverPath)
					saveText =  '   Saved Results: {} - Saved'.format(bestValue)
					count = 1
			elif bestValue == testStepArray[iteration,episode]:
				count += 1
				if count >= maxCount:
					saver.save(sess, saverPath)
					saveText =  '   Saved Results: {} - Saved'.format(bestValue)
				else:
					saveText = '   Saved Results: {} - Stopped'.format(bestValue)
			elif testStepArray[iteration,episode] <= bestValue - 50:
					count = 0
			maxCount = max(count,maxCount)

			
			if currentSystem == 'Windows':
				os.system('cls')
			else:
				os.system('clear')

			timePerEpisode = time() - initialTime
			print('Iteration : {}  \nEpisode : {}    Reward : {:.5f}    Loss : {:.5f}     Bellman-Loss : {:.5f}  \nTrain-Steps : {}   Test-Steps : {}  Time per episode : {:.1f}s  Longest Streak : {}'.format(iteration, episode, rewardArray[iteration,episode],lossArray[iteration,episode],bellmanLossArray[iteration,episode],stepArray[iteration,episode],testStepArray[iteration,episode],timePerEpisode, maxCount) + saveText)
		

	np.save(numpyFolderPath + 'bellmanLossArray_{}'.format(learningRate), bellmanLossArray)
	np.save(numpyFolderPath + 'lossArray_{}'.format(learningRate), lossArray)
	np.save(numpyFolderPath + 'rewardArray_{}'.format(learningRate), rewardArray)
	np.save(numpyFolderPath + 'train_stepArray_{}'.format(learningRate), stepArray)
	np.save(numpyFolderPath + 'test_stepArray_{}'.format(learningRate), testStepArray)



savePlots(np.mean(bellmanLossArray, axis = 0), plotFolderPath + 'bellmanLossPlot_{}'.format(learningRate), 'bellman loss', mode = '')
savePlots(np.mean(lossArray, axis = 0), plotFolderPath + 'lossPlot_{}'.format(learningRate), 'loss', mode = '')
savePlots(np.mean(rewardArray, axis = 0), plotFolderPath + 'rewardPlot_{}'.format(learningRate), 'cumulative reward', mode = 'reward')
savePlots(np.mean(stepArray, axis = 0), plotFolderPath + 'train_stepPlot_{}'.format(learningRate), 'train steps before termination', mode = 'step')
savePlots(np.mean(testStepArray, axis = 0), plotFolderPath + 'test_stepPlot_{}'.format(learningRate), 'test steps before termination', mode = 'step')


