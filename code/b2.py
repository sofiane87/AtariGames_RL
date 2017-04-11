import gym
import tensorflow as tf
import numpy as np
from PIL import Image
import platform
from time import time
import os
import sys
import platform
from time import time
currentSystem = platform.system()


############################## PARAMETERS ######################
gameName = ['Pong-v3','MsPacman-v3','Boxing-v3']
gameIndex = 1
argList = sys.argv[1:]
if ('-game' in argList):
	gameIndex = int(argList[argList.index('-game')+1])
game = gameName[gameIndex]


discount = 0.99
epsilon = 0.1
actionDim = gym.make(game).action_space.n
bufferLength = 4
nEpisodesToTrain = 10**6
numberOfEpisodes = 100
pictureSize = 28
useBias = False
learningRate = 10**-5
env = gym.make(game)


###############################################################

def clipReward(reward):
	if reward <= -1:
		return -1
	elif reward >= 1:
		return 1
	else :
		return 0


def preprocessImageStack(np_arr):
	#stacked greyscale images
	#convert to grayscale
	img = Image.fromarray(np_arr, 'RGB').convert('L')
	#resize
	img = img.resize((28, 28), Image.ANTIALIAS)

	stacked_np = np.array(img).astype(np.uint8)
	return stacked_np


def runGameGreedy(sess,decisionVector, env, numberOfEpisodes, stackSize):
	actionSpace = env.action_space.n
	numberOfFrames = []
	cumulativeScores = []
	for episode in range(numberOfEpisodes):
		Done = False
		observation = env.reset()

		observationList = np.zeros([28,28,4])
		for i in range(4):
			observationList[:,:,i] = preprocessImageStack(observation)
		
		observationBuffer = np.array([observationList])	
		
		index = 0
		tempScore = 0
		while not(Done):
			## preprocessing
			feedDict = {
				currentState : observationBuffer
			}
			action = np.argmax(sess.run(decisionVector, feed_dict = feedDict))
			observation, reward, Done, info = env.step(action)
			tempScore += clipReward(reward) * discount**index
			index += 1
			observationList[:,:,:-1] = observationList[:,:,1:]
			observationList[:,:,-1] = preprocessImageStack(observation)
			observationBuffer = np.array([observationList])	

		cumulativeScores.append(tempScore)
		#numberOfFrames.append(index//4)
		numberOfFrames.append(index)
		print('Episode : {}  Frame Count : {} Score : {}'.format(episode, index, tempScore), end = '\r')

	frameMean = np.mean(numberOfFrames)
	frameStd = np.std(numberOfFrames)
	scoreMean = np.mean(cumulativeScores)
	scoreStd = np.std(cumulativeScores)
		

	print('Mean Frame Count : {} -  Frame Count STD {}'.format(frameMean,frameStd) )
	print('Mean Score : {} -  Score STD {}'.format(scoreMean,scoreStd) )

	return frameMean, frameMean, scoreMean, scoreMean


def updateParams():
	for x in tf.global_variables():
		if ('copy-' in x.name):
			for y in tf.global_variables():
				if (y.name == x.name[5:]):
					x = y
	return tf.Variable(1)


def buildNetwork(stackSize= bufferLength, pictureSize = pictureSize, stride = 2, useBias = useBias, discountFactor = discount, learningRate = learningRate):	
	
	currentState = tf.placeholder(tf.float32, [None, pictureSize, pictureSize, stackSize])

	# Output Hidden Layer
	convWeight1 = tf.Variable(tf.truncated_normal([6,6, 4, 16], stddev = 0.01), name = 'conv1Weight')
	convLayer1 = tf.nn.conv2d(currentState, convWeight1 , strides = [1, stride, stride, 1], padding = 'SAME')
	reluConv1 = tf.nn.relu(convLayer1)

	convWeight2 = tf.Variable(tf.truncated_normal([4,4, 16, 32], stddev = 0.01), name = 'conv2Weight')
	convLayer2 = tf.nn.conv2d(reluConv1, convWeight2, strides = [1, stride, stride, 1], padding = 'SAME')
	reluConv2 = tf.nn.relu(convLayer2)



	output_shape = int(reluConv2.get_shape()[1]*reluConv2.get_shape()[2]*reluConv2.get_shape()[3])
	flatCurrentQ = tf.reshape(reluConv2,shape=[-1,1568]) 
	hiddenWeights = tf.Variable(tf.truncated_normal([output_shape,256], stddev = 0.01), name = 'hiddenWeight')
	if useBias:
		hiddenBias = tf.Variable(tf.truncated_normal([256], stddev = 0.01), name = 'hiddenBias')
		hiddenCurrentQ = tf.nn.relu(tf.matmul(flatCurrentQ,hiddenWeights) + hiddenBias)
	else:
		hiddenCurrentQ = tf.nn.relu(tf.matmul(flatCurrentQ,hiddenWeights))

	# Output Q matrix for all actions 
	finalWeights = tf.Variable(tf.truncated_normal([256,actionDim], stddev = 0.01), name = 'finalWeight')
	if useBias:
		finalBias = tf.Variable(tf.truncated_normal([actionDim], stddev = 0.01), name = 'finalBias')
		qMatrix = tf.matmul(hiddenCurrentQ,finalWeights) + finalBias
	else:
		qMatrix = tf.matmul(hiddenCurrentQ,finalWeights)

	# Picking the proper QValues 

	nextState = tf.placeholder(tf.float32, [None, pictureSize, pictureSize, stackSize])

	# Output Hidden Layer
	copyConvWeight1 = tf.Variable(tf.truncated_normal([6,6, 4, 16], stddev = 0.01), name = 'copy-conv1Weight')
	copyConvLayer1 = tf.nn.conv2d(nextState, copyConvWeight1 , strides = [1, stride, stride, 1], padding = 'SAME')
	copyReluConv1 = tf.nn.relu(copyConvLayer1)

	copyConvWeight2 = tf.Variable(tf.truncated_normal([4,4, 16, 32], stddev = 0.01), name = 'copy-conv2Weight')
	copyConvLayer2 = tf.nn.conv2d(copyReluConv1, copyConvWeight2, strides = [1, stride, stride, 1], padding = 'SAME')
	reluConv2 = tf.nn.relu(copyConvLayer2)

	flatnextQ = tf.reshape(reluConv2,shape=[-1,output_shape]) 
	copyHiddenWeights = tf.Variable(tf.truncated_normal([output_shape,256], stddev = 0.01), name = 'copy-hiddenWeight')
	if useBias:
		copyHiddenBias = tf.Variable(tf.truncated_normal([256], stddev = 0.01), name = 'copy-hiddenBias')
		hiddennextQ = tf.nn.relu(tf.matmul(flatnextQ,copyHiddenWeights) + copyHiddenBias)
	else:
		hiddennextQ = tf.nn.relu(tf.matmul(flatnextQ,copyHiddenWeights))

	# Output Q matrix for all actions 
	copyFinalWeights = tf.Variable(tf.truncated_normal([256,actionDim], stddev = 0.01), name = 'copy-finalWeight')
	if useBias:
		copyFinalBias = tf.Variable(tf.truncated_normal([actionDim], stddev = 0.01), name = 'copy-finalBias')
		nextQ = tf.matmul(hiddennextQ,copyFinalWeights) + copyFinalBias
	else:
		nextQ = tf.matmul(hiddennextQ,copyFinalWeights)

	########### UPDATE OPERATIONS ################

	convWeight1AssignOp = copyConvWeight1.assign(convWeight1) 
	convWeight2AssignOp = copyConvWeight2.assign(convWeight2) 
	hiddenWeightsAssignOp = copyHiddenWeights.assign(hiddenWeights) 
	finalWeightsAssignOp = copyFinalWeights.assign(finalWeights) 
	if useBias:
		finalBiasAssignOp = copyFinalBias.assign(finalBias) 
		hiddenBiasAssignOp = copyNiddenBias.assign(hiddenBias) 

	##################################

	# Picking the proper QValues 
	currentAction = tf.placeholder(tf.int32, [None])
	rowIndices = tf.placeholder(tf.int32, [None])
	fullIndices = tf.stack([rowIndices, currentAction], axis = 1)
	predictedQValues = tf.gather_nd(qMatrix, fullIndices)

	# Computing the target
	reward_value = tf.placeholder(tf.float32, [None])
	target_value = reward_value + tf.stop_gradient(discountFactor*tf.reduce_max(nextQ, axis =1))

	########################### LOSS ###########################
	loss = 0.5*tf.reduce_mean(tf.square(target_value - predictedQValues))
	bellman_loss = tf.reduce_mean(target_value - predictedQValues)

	# Train Step 
	train_step = tf.train.RMSPropOptimizer(learningRate).minimize(loss)

	if useBias:
		return currentState, nextState, reward_value, currentAction, rowIndices, loss, bellman_loss, train_step, qMatrix, convWeight1AssignOp,convWeight2AssignOp,hiddenWeightsAssignOp,finalWeightsAssignOp, finalBiasAssignOp,hiddenBiasAssignOp
	else:
		return currentState, nextState, reward_value, currentAction, rowIndices, loss, bellman_loss, train_step, qMatrix, convWeight1AssignOp,convWeight2AssignOp,hiddenWeightsAssignOp,finalWeightsAssignOp


def donotupdate():
	return tf.Variable(1)

######################################################################################

currentState, nextState, reward_value, currentAction, rowIndices, loss, bellman_loss, train_step, qMatrix, convWeight1AssignOp,convWeight2AssignOp,hiddenWeightsAssignOp,finalWeightsAssignOp = buildNetwork()


with tf.Session() as sess :
	init = tf.global_variables_initializer()
	sess.run(init)
	runGameGreedy(sess,qMatrix, env, numberOfEpisodes, bufferLength)