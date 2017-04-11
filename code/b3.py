import gym
import tensorflow as tf
import numpy as np
from time import time
from PIL import Image
import matplotlib.pyplot as plt
import sys
import platform
from time import time
import os
currentSystem = platform.system()
if currentSystem == 'Windows':
	import msvcrt # built-in module
else:
	import select



########################## Global Variables ###########################

global firstRun 
global bufferSize 
global bufferObservations 
global bufferNextObservations 
global bufferActions 
global bufferRewards 
global BatchSize 
global testSteps
global actionDim
global render_option
global real_time_render


###########################  Option Parameters ###########################

firstRun = False
real_time_render = True
render_option = False

#######################################################################
gameName = ['Pong-v3','MsPacman-v3','Boxing-v3']
gameIndex = 1
argList = sys.argv[1:]
if ('-game' in argList):
	gameIndex = int(argList[argList.index('-game')+1])
game = gameName[gameIndex]
env = gym.make(game)
continueTraining = False
if ('-continue' in argList):
	continueTraining = True


################ Assignment Parameters #############################

discount = 0.99
epsilon = 0.1
bufferSize = 100000
numberOfEvalEpisodes = 100
nSteps = 10**6
nEpisodes = int(nSteps/100)
numberOfEpisodesToTest = 20
testSteps = 50000
BatchSize = 32
updateRate = 5000


###################### Network Parameter #########################
actionDim = env.action_space.n
stackSize = 4
pictureSize = 60
if pictureSize != 28:
	pictureString = 'size_{}_'.format(pictureSize)
else:
	pictureString = ''

stride = 2
useBias = False
discountFactor = discount
learningRate = 0.0001
std = learningRate/10
###########################  Buffer Initialization ###########################


bufferObservations = np.zeros([bufferSize,  pictureSize, pictureSize, stackSize]).astype(np.uint8)
bufferNextObservations = np.zeros([bufferSize,  pictureSize, pictureSize, stackSize]).astype(np.uint8)
bufferActions = np.zeros([bufferSize])
bufferRewards = np.zeros([bufferSize])
testIndexes = np.zeros([nSteps//testSteps])

############################  PATHS Parameters ###########################

plotFolderPath = 'Problem-B' + os.path.sep +  'Q3' + os.path.sep + game + os.path.sep + 'Plot' + os.path.sep + pictureString
numpyFolderPath = 'Problem-B' + os.path.sep + 'Q3' + os.path.sep + game + os.path.sep +  'Numpy' + os.path.sep + pictureString
modelFolderPath = 'Problem-B' + os.path.sep + 'Q3' + os.path.sep + game + os.path.sep +  'Model' + os.path.sep + pictureString

##########################################################################


if not(os.path.exists(plotFolderPath)):
	os.makedirs(plotFolderPath)

if not(os.path.exists(numpyFolderPath)):
	os.makedirs(numpyFolderPath)

if not(os.path.exists(modelFolderPath)):
	os.makedirs(modelFolderPath)


##########################################################################

def update_render_option():
	global render_option
	global real_time_render
	if currentSystem == 'Windows':
		if msvcrt.kbhit():
			currentKey = chr(msvcrt.getch()[0]).lower()
			#print('Current Key : {}'.format(currentKey))
			if currentKey == 'r':
					render_option = not(render_option)
					print('render_option Changed To : {}'.format(render_option))
					if not(render_option):
						env.render(close=True)
			elif currentKey == 's':
					real_time_render = False
					print('real_time_render Changed To : {}'.format(real_time_render))
	else:
		inputs,_,_ = select.select([sys.stdin],[],[],0)
		for s in inputs:
			if s == sys.stdin:
				currentKey = str(sys.stdin.readline())[0].lower()
				#print('Current Key : {}'.format(currentKey))
				if currentKey == 'r':
					render_option = not(render_option)
					print('render_option Changed To : {}'.format(render_option))
					if not(render_option):
						env.render(close=True)
				elif currentKey == 's':
					real_time_render = False
					print('real_time_render Changed To : {}'.format(real_time_render))



def clipReward(reward):
	if reward <= -1:
		return -1
	elif reward >= 1:
		return 1
	else :
		return 0


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




def preprocessImageStack(np_arr):
	#stacked greyscale images
	#convert to grayscale
	img = Image.fromarray(np_arr, 'RGB').convert('L')
	#resize
	img = img.resize((pictureSize, pictureSize), Image.ANTIALIAS)

	stacked_np = np.array(img).astype(np.uint8)
	return stacked_np


def runGameGreedy(env, sess,decisionVector, game, numberOfEpisodes, stackSize):
	global render_option
	global real_time_render

	actionSpace = env.action_space.n
	numberOfFrames = []
	cumulativeScores = []
	print('--------------- Starting Testing ---------------')
	for episode in range(numberOfEpisodes):
		Done = False
		observation = env.reset()

		observationBuffer = np.zeros([1,pictureSize,pictureSize,4])
		for i in range(4):
			observationBuffer[:,:,:,i] = preprocessImageStack(observation)
				
		index = 0
		tempScore = 0
		while not(Done):
			if real_time_render:
					update_render_option()
			if render_option :
				env.render()


			feedDict = {
				currentState : observationBuffer
			}
			action = np.argmax(sess.run(decisionVector, feed_dict = feedDict))
			observation, reward, Done, info = env.step(action)
			tempScore += clipReward(reward) * discount**index
			index += 1
			observationBuffer[:,:,:,:-1] = observationBuffer[:,:,:,1:]
			observationBuffer[:,:,:,-1] = preprocessImageStack(observation)
		

		print('Episode : {}  Frame Count : {} Score : {}'.format(episode, index, tempScore))


		cumulativeScores.append(tempScore)
		numberOfFrames.append(index)

	frameMean = np.mean(numberOfFrames)
	frameStd = np.std(numberOfFrames)
	scoreMean = np.mean(cumulativeScores)
	scoreStd = np.std(cumulativeScores)
		

	return frameMean, frameStd, scoreMean, scoreStd



def populateBuffer(env, game, numberOfSteps, stackSize):
	global bufferObservations 
	global bufferNextObservations 
	global bufferActions 
	global bufferRewards
	global render_option
	global real_time_render

	localcounter = 0
	actionSpace = env.action_space.n
	done = True
	observation = env.reset()

	while localcounter != numberOfSteps:
		if real_time_render:
			update_render_option()
		if render_option :
			env.render()
		if done:
			observation = env.reset()
			observationBuffer = np.zeros([pictureSize,pictureSize,stackSize])
			for i in range(stackSize):
				observationBuffer[:,:,i] = preprocessImageStack(observation)

		action = np.random.randint(0,actionSpace)
		observation, reward, done, _ = env.step(action)

		bufferObservations[localcounter] = observationBuffer
		bufferActions[localcounter] = action
		bufferRewards[localcounter] = reward
		observationBuffer[:,:,:-1] = observationBuffer[:,:,1:]
		observationBuffer[:,:,-1] =  preprocessImageStack(observation)
		if done : 
			observationBuffer = np.zeros([pictureSize,pictureSize,4])
		bufferNextObservations[localcounter] = observationBuffer
		
		localcounter += 1
		if ((localcounter % (numberOfSteps//100)) == 0):
			if currentSystem == 'Windows':
				os.system('cls')
			else:
				os.system('clear')
			print('populating ongoing : {}% done'.format((localcounter*100)//(numberOfSteps)))
		


def greedy_action(observation, sess, qVector):
	return np.argmax(sess.run(qVector, feed_dict = {currentState : observation}))






def trainAnEpisode(env, sess, loss_var, bellman_loss_var, train_op, qVector, fedcounter, currentIndex):

	global firstRun 
	global bufferSize 
	global bufferObservations 
	global bufferNextObservations 
	global bufferActions 
	global bufferRewards 
	global BatchSize 
	global testSteps
	global actionDim
	global render_option
	global real_time_render

	#print('EPISODE : ', i_episode)
	observation = env.reset()
	
	loss = 0
	bellmanLoss = 0
	done = False
	beginingOfEpisode = True
	t = 0
	reward = 0


	currentObservation = preprocessImageStack(observation) 
	currentObservationBuffer = np.zeros([1,pictureSize,pictureSize,4])
	for i in range(4):
		currentObservationBuffer[:,:,:,i] = currentObservation	
	updatedString = ''

	while not(done):		
		if fedcounter % updateRate == 0:
			sess.run([convWeight1AssignOp,convWeight2AssignOp, hiddenWeightsAssignOp, finalWeightsAssignOp])
			if useBias:
				sess.run([finalBiasAssignOp, hiddenBiasAssignOp])

			updatedString = ' - updated'
	
		if real_time_render:
				update_render_option()
		if render_option :
			env.render()


		randValue = np.random.random(1)
		if randValue <= epsilon:
			action = np.random.randint(0,high = actionDim)
		else:
			action = greedy_action(currentObservationBuffer, sess, qVector)


		observation, env_reward, done, info = env.step(action)
		

		currentObservation = preprocessImageStack(observation) 

		tempreward = clipReward(env_reward)
		reward += (discountFactor)**(t) * tempreward
		
		if done:
			numberOfFrames = t +1 


		if firstRun:
			bufferObservations[:] = currentObservationBuffer[0]
			bufferActions[:] = action
			bufferRewards[:] = tempreward
			currentObservationBuffer[:,:,:,:-1] = currentObservationBuffer[:,:,:,1:]
			currentObservationBuffer[:,:,:,-1] = currentObservation
			if done : 
				currentObservationBuffer[:,:,:,:] = np.zeros([1,pictureSize,pictureSize,4])
			bufferNextObservations[:] = currentObservationBuffer[0]
			firstRun = False
		else:
			bufferObservations[currentIndex] = currentObservationBuffer[0]
			bufferActions[currentIndex] = action
			bufferRewards[currentIndex] = tempreward
			currentObservationBuffer[:,:,:,:-1] = currentObservationBuffer[:,:,:,1:]
			currentObservationBuffer[:,:,:,-1] = currentObservation
			if done : 
				currentObservationBuffer[:,:,:,:] = np.zeros([1,pictureSize,pictureSize,4])
			bufferNextObservations[currentIndex] = currentObservationBuffer[0]

		############ BATCHING #########################
		
		batchIndexes = np.random.randint(0, high=bufferSize, size=[BatchSize])
		observationBatch = np.take(bufferObservations, batchIndexes, axis = 0)
		actionBatch = np.take(bufferActions, batchIndexes, axis = 0)
		nextobservationBatch = np.take(bufferNextObservations, batchIndexes, axis = 0)
		rewardBatch = np.take(bufferRewards, batchIndexes, axis = 0)
		indiceBatch = np.arange(BatchSize)


		feed_dict = {currentState : observationBatch, 
					 currentAction : actionBatch, 
					 rowIndices : indiceBatch, 
					 nextState : nextobservationBatch, 
					 reward_value : rewardBatch,
					 }

		temploss, tempBellmanLoss, _ = sess.run([loss_var,bellman_loss_var, train_op], feed_dict=feed_dict)

		loss += temploss 
		bellmanLoss += tempBellmanLoss
		currentIndex += 1
		currentIndex = currentIndex % bufferSize
		t += 1
		fedcounter += 1
		# listAppendi
	loss = loss/t
	bellmanLoss = bellmanLoss/t


	return currentIndex, loss, bellmanLoss, reward, numberOfFrames, updatedString




######################## NEURAL NETWORK #######################
	
def buildNetwork(stackSize= stackSize,pictureSize = pictureSize, stride = 2, useBias = False, discountFactor = discount, learningRate = learningRate):	
	
	currentState = tf.placeholder(tf.float32, [None, pictureSize, pictureSize, stackSize])

	# Output Hidden Layer
	convWeight1 = tf.Variable(tf.truncated_normal([6, 6, 4, 16], stddev = std), name = 'conv1Weight')
	convLayer1 = tf.nn.conv2d(currentState, convWeight1 , strides = [1, stride, stride, 1], padding = 'SAME')
	reluConv1 = tf.nn.relu(convLayer1)

	convWeight2 = tf.Variable(tf.truncated_normal([4,4, 16, 32], stddev = std), name = 'conv2Weight')
	convLayer2 = tf.nn.conv2d(reluConv1, convWeight2, strides = [1, stride, stride, 1], padding = 'SAME')
	reluConv2 = tf.nn.relu(convLayer2)



	output_shape = int(reluConv2.get_shape()[1]*reluConv2.get_shape()[2]*reluConv2.get_shape()[3])
	flatCurrentQ = tf.reshape(reluConv2,shape=[-1,output_shape]) 
	hiddenWeights = tf.Variable(tf.truncated_normal([output_shape,256], stddev = std), name = 'hiddenWeight')
	if useBias:
		hiddenBias = tf.Variable(tf.truncated_normal([256], stddev = std), name = 'hiddenBias')
		hiddenCurrentQ = tf.nn.relu(tf.matmul(flatCurrentQ,hiddenWeights) + hiddenBias)
	else:
		hiddenCurrentQ = tf.nn.relu(tf.matmul(flatCurrentQ,hiddenWeights))

	# Output Q matrix for all actions 
	finalWeights = tf.Variable(tf.random_normal([256,actionDim], stddev = std), name = 'finalWeight')
	if useBias:
		finalBias = tf.Variable(tf.random_normal([actionDim], stddev = std), name = 'finalBias')
		qMatrix = tf.matmul(hiddenCurrentQ,finalWeights) + finalBias
	else:
		qMatrix = tf.matmul(hiddenCurrentQ,finalWeights)

	# Picking the proper QValues 

	nextState = tf.placeholder(tf.float32, [None, pictureSize, pictureSize, stackSize])

	# Output Hidden Layer
	copyConvWeight1 = tf.Variable(tf.truncated_normal([6,6, 4, 16], stddev = std), name = 'copy-conv1Weight')
	copyConvLayer1 = tf.nn.conv2d(nextState, copyConvWeight1 , strides = [1, stride, stride, 1], padding = 'SAME')
	copyReluConv1 = tf.nn.relu(copyConvLayer1)

	copyConvWeight2 = tf.Variable(tf.truncated_normal([4,4, 16, 32], stddev = std), name = 'copy-conv2Weight')
	copyConvLayer2 = tf.nn.conv2d(copyReluConv1, copyConvWeight2, strides = [1, stride, stride, 1], padding = 'SAME')
	reluConv2 = tf.nn.relu(copyConvLayer2)

	flatnextQ = tf.reshape(reluConv2,shape=[-1,output_shape]) 
	copyHiddenWeights = tf.Variable(tf.truncated_normal([output_shape,256], stddev = std), name = 'copy-hiddenWeight')
	if useBias:
		copyHiddenBias = tf.Variable(tf.truncated_normal([256], stddev = std), name = 'copy-hiddenBias')
		hiddennextQ = tf.nn.relu(tf.matmul(flatnextQ,copyHiddenWeights) + copyHiddenBias)
	else:
		hiddennextQ = tf.nn.relu(tf.matmul(flatnextQ,copyHiddenWeights))

	# Output Q matrix for all actions 
	copyFinalWeights = tf.Variable(tf.random_normal([256,actionDim], stddev = std), name = 'copy-finalWeight')
	if useBias:
		copyFinalBias = tf.Variable(tf.random_normal([actionDim], stddev = std), name = 'copy-finalBias')
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



#################################  INITIALIZING NETWORK  ###################################


currentState, nextState, reward_value, currentAction, rowIndices, loss, bellman_loss, train_step, qMatrix, convWeight1AssignOp,convWeight2AssignOp,hiddenWeightsAssignOp,finalWeightsAssignOp = buildNetwork()


#################################  DEFINING NUMPY ARRAYS  ##################################


bellmanLossArray = np.zeros([nEpisodes])
lossArray = np.zeros([nEpisodes])
numberOfFrames = np.zeros([nEpisodes])
testFrameMean = np.zeros([nSteps//testSteps])
trainScore = np.zeros([nEpisodes])
testScoreMean = np.zeros([nSteps//testSteps])


#################################  INITIZATING SESSION  ##################################

if not(continueTraining):
	populateBuffer(env, game, bufferSize, stackSize)

with tf.Session() as sess :
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	sess.run(init)


	currentIndex = 0
	counter = 0
	episode = 0

	bestValue = -10^10
	saverPath = modelFolderPath + 'Model_{}.ckpt'.format(learningRate)

	if continueTraining:
		saver.restore(sess, saverPath)
		
		bellmanLossArray = np.load(numpyFolderPath + 'bellmanLossArray_{}'.format(learningRate) + '.npy')
		lossArray = np.load(numpyFolderPath + 'lossArray_{}'.format(learningRate) + '.npy')
		trainScore = np.load(numpyFolderPath + 'trainScore_{}'.format(learningRate) + '.npy')
		testScoreMean = np.load(numpyFolderPath + 'testScoreMean_{}'.format(learningRate) + '.npy')
		numberOfFrames = np.load(numpyFolderPath + 'numberOfFrames_{}'.format(learningRate) + '.npy')
		testFrameMean = np.load(numpyFolderPath + 'testFrameMean_{}'.format(learningRate) + '.npy')

		bufferObservations = np.load(numpyFolderPath + 'bufferObservations_{}'.format(learningRate) + '.npy')
		bufferNextObservations = np.load(numpyFolderPath + 'bufferNextObservations_{}'.format(learningRate) + '.npy')
		bufferActions = np.load(numpyFolderPath + 'bufferActions_{}'.format(learningRate) + '.npy')
		bufferRewards = np.load(numpyFolderPath + 'bufferRewards_{}'.format(learningRate) + '.npy')

		counter = int(numberOfFrames.sum())
		episode = np.array(numberOfFrames != 0).astype(int).sum()
		bestValue = testScoreMean.max()
		currentIndex = currentIndex % bufferSize




		print('initializing at {} episodes or {} steps '.format(counter,episode))
		print('best Value found : {}'.format(bestValue))


	while counter <= nSteps:
		beginingOfEpisode = time()
		

		##################################### TESTING ######################
		if testIndexes[counter//testSteps] == 0:
			testFrameMean[counter//testSteps], _ , testScoreMean[counter//testSteps], _ = runGameGreedy(env, sess,qMatrix, game, 100, stackSize)
			
			if bestValue <= testScoreMean[counter//testSteps]:
				bestValue = testScoreMean[counter//testSteps]
				saver.save(sess, saverPath)

			np.save(numpyFolderPath + 'bellmanLossArray_{}'.format(learningRate), bellmanLossArray)
			np.save(numpyFolderPath + 'lossArray_{}'.format(learningRate), lossArray)
			np.save(numpyFolderPath + 'trainScore_{}'.format(learningRate), trainScore)
			np.save(numpyFolderPath + 'testScoreMean_{}'.format(learningRate), testScoreMean)
			np.save(numpyFolderPath + 'numberOfFrames_{}'.format(learningRate), numberOfFrames)
			np.save(numpyFolderPath + 'testFrameMean_{}'.format(learningRate), testFrameMean)


			np.save(numpyFolderPath + 'bufferObservations_{}'.format(learningRate), bufferObservations)
			np.save(numpyFolderPath + 'bufferNextObservations_{}'.format(learningRate), bufferNextObservations)
			np.save(numpyFolderPath + 'bufferActions_{}'.format(learningRate), bufferActions)
			np.save(numpyFolderPath + 'bufferRewards_{}'.format(learningRate), bufferRewards)


			testIndexes[counter//testSteps] = 1
		

		##########################################################################

		currentIndex, lossArray[episode], bellmanLossArray[episode], trainScore[episode], numberOfFrames[episode], updatedString = trainAnEpisode(env, sess, loss, bellman_loss, train_step, qMatrix, counter, currentIndex)		

		tempCounter = counter
		counter += int(numberOfFrames[episode])

	#########################################################################
			
		if currentSystem == 'Windows':
			os.system('cls')
		else:
			os.system('clear')

		timePerEpisode = time() - beginingOfEpisode
		print('counter : {}  current-Index : {} Episode : {}    Score : {:.5f}    Loss : {:.5f}     Bellman-Loss : {:.5f}  \nTrain-Steps : {}   Test Frame Count : {:.1f}   Test Score : {:.2f}  Time per Episode : {:.1f}'.format(counter, currentIndex, episode, trainScore[episode],lossArray[episode],bellmanLossArray[episode],numberOfFrames[episode],testFrameMean[tempCounter//testSteps], testScoreMean[tempCounter//testSteps], timePerEpisode) + updatedString)

		episode += 1

############# CUTTING ARRAYS ##################

bellmanLossArray = bellmanLossArray[:episode] 
lossArray = lossArray[:episode] 
numberOfFrames = numberOfFrames[:episode] 
testFrameMean = testFrameMean[:episode] 
trainScore = trainScore[:episode] 
testScoreMean = testScoreMean[:episode] 

############# SAVING ARRAYS ##################

np.save(numpyFolderPath + 'bellmanLossArray_{}'.format(learningRate), bellmanLossArray)
np.save(numpyFolderPath + 'lossArray_{}'.format(learningRate), lossArray)
np.save(numpyFolderPath + 'trainScore_{}'.format(learningRate), trainScore)
np.save(numpyFolderPath + 'testScoreMean_{}'.format(learningRate), testScoreMean)
np.save(numpyFolderPath + 'numberOfFrames_{}'.format(learningRate), numberOfFrames)
np.save(numpyFolderPath + 'testFrameMean_{}'.format(learningRate), testFrameMean)

############# PLOTTING ##################


savePlots(np.mean(bellmanLossArray, axis = 0), plotFolderPath + 'bellmanLossPlot_{}'.format(learningRate), 'bellman loss', mode = '')
savePlots(np.mean(lossArray, axis = 0), plotFolderPath + 'lossPlot_{}'.format(learningRate), 'loss', mode = '')
savePlots(np.mean(trainScore, axis = 0), plotFolderPath + 'trainScore_{}'.format(learningRate), 'cumulative Score', mode = '')
savePlots(np.mean(testScoreMean, axis = 0), plotFolderPath + 'testScoreMean_{}'.format(learningRate), 'Average cumulative Test Score', mode = '')
savePlots(np.mean(numberOfFrames, axis = 0), plotFolderPath + 'numberOfFrames_{}'.format(learningRate), 'train frame number before termination', mode = '')
savePlots(np.mean(testFrameMean, axis = 0), plotFolderPath + 'testFrameMean_{}'.format(learningRate), 'test frame number before termination', mode = '')
