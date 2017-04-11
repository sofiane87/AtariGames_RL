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

global discountFactor
global real_time_render
global render_option

real_time_render = True
render_option = False
discountFactor = 0.99

def clipReward(reward):
	if reward <= -1:
		return -1
	elif reward >= 1:
		return 1
	else :
		return 0

def preprocessImageStack(np_arr,pictureSize):
	#stacked greyscale images
	#convert to grayscale
	img = Image.fromarray(np_arr, 'RGB').convert('L')
	#resize
	img = img.resize((pictureSize, pictureSize), Image.ANTIALIAS)

	stacked_np = np.array(img).astype(np.uint8)
	return stacked_np


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





def buildModel3_1(std):


	currentState = tf.placeholder(tf.float32, [None, 4])

	# Output Q matrix for all actions 
	qWeight = tf.Variable(tf.random_normal([4, 2], stddev=std))
	qMatrix = tf.matmul(currentState, qWeight)


	return currentState, qMatrix


def buildModel3_2():

	currentState = tf.placeholder(tf.float32, [None, 4])

	# Output Hidden Layer
	qHiddenWeight = tf.Variable(tf.random_normal([4, 100], stddev=0.1))
	qHiddenMatrix = tf.nn.relu(tf.matmul(currentState, qHiddenWeight))

	# Output Q matrix for all actions 
	qWeight = tf.Variable(tf.random_normal([100, 2], stddev=0.1))
	qMatrix = tf.matmul(qHiddenMatrix, qWeight)

	return currentState, qMatrix


def buildModel4To6(hiddenSize,std):

	currentState = tf.placeholder(tf.float32, [None, 4])

	# Output Hidden Layer
	qHiddenWeight = tf.Variable(tf.random_normal([4, hiddenSize], stddev=std))
	qHiddenMatrix = tf.nn.relu(tf.matmul(currentState, qHiddenWeight))


	# Output Q matrix for all actions 
	qWeight = tf.Variable(tf.random_normal([hiddenSize, 2], stddev=std))
	qMatrix = tf.matmul(qHiddenMatrix, qWeight)

	
	return currentState, qMatrix
	#################################### GENERATING THE DATA ######################################

def buildModel7(hiddenSize,std):

	currentState = tf.placeholder(tf.float32, [None, 4])

	# Output Hidden Layer
	qHiddenWeight = tf.Variable(tf.random_normal([4, hiddenSize], stddev=std))
	qHiddenMatrix = tf.nn.relu(tf.matmul(currentState, qHiddenWeight))

	qWeight = tf.Variable(tf.random_normal([hiddenSize, 2], stddev=std))
	qMatrix = tf.matmul(qHiddenMatrix, qWeight)


	# Output Hidden Layer
	copyQHiddenWeight = tf.Variable(tf.random_normal([4, hiddenSize], stddev=std))
	copyQWeight = tf.Variable(tf.random_normal([hiddenSize, 2], stddev=std))

	return currentState, qMatrix


def buildModel8(hiddenSize,std):


	# Output Hidden Layer
	firstHiddenWeight = tf.Variable(tf.random_normal([4, hiddenSize], stddev=std))
	secondHiddenWeight = tf.Variable(tf.random_normal([4, hiddenSize], stddev=std))
	firstFinalWeight = tf.Variable(tf.random_normal([hiddenSize, 2], stddev=std))
	secondFinalWeight = tf.Variable(tf.random_normal([hiddenSize, 2], stddev=std))


	evaluatingHiddenWeight = firstHiddenWeight
	selectionHiddenWeight = secondHiddenWeight

	evaluatingFinalWeight = secondFinalWeight
	selectionFinalWeight = firstFinalWeight


	currentState = tf.placeholder(tf.float32, [None, 4])
	is_training_bool = tf.placeholder(tf.bool)


	mainQHiddenMatrix = tf.nn.relu(tf.matmul(currentState, evaluatingHiddenWeight))
	secondmainQHiddenMatrix = tf.nn.relu(tf.matmul(currentState, selectionHiddenWeight))


	mainQMatrix = tf.matmul(mainQHiddenMatrix, evaluatingFinalWeight)
	secondMainMatrix = tf.matmul(secondmainQHiddenMatrix, selectionFinalWeight)


	averageQMatrix = 1/2 * mainQMatrix + 1/2 * secondMainMatrix

	return currentState, averageQMatrix

	
def buildModelB3(actionDim, stackSize= 4,pictureSize = 28, stride = 2, std = 0.01):	
	
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
	hiddenCurrentQ = tf.nn.relu(tf.matmul(flatCurrentQ,hiddenWeights))

	# Output Q matrix for all actions 
	finalWeights = tf.Variable(tf.random_normal([256,actionDim], stddev = std), name = 'finalWeight')
	qMatrix = tf.matmul(hiddenCurrentQ,finalWeights)

	# Picking the proper QValues 


	# Output Hidden Layer
	copyConvWeight1 = tf.Variable(tf.truncated_normal([6,6, 4, 16], stddev = std), name = 'copy-conv1Weight')
	copyConvWeight2 = tf.Variable(tf.truncated_normal([4,4, 16, 32], stddev = std), name = 'copy-conv2Weight')
	copyHiddenWeights = tf.Variable(tf.truncated_normal([output_shape,256], stddev = std), name = 'copy-hiddenWeight')

	# Output Q matrix for all actions 
	copyFinalWeights = tf.Variable(tf.random_normal([256,actionDim], stddev = std), name = 'copy-finalWeight')



	return currentState, qMatrix


def greedy_action(currentState, observation, sess, qVector):
	return np.argmax(sess.run(qVector, feed_dict = {currentState : observation}))


def runGreedyGame(env, sess,decisionVector, game, numberOfEpisodes, stackSize, pictureSize):
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
			observationBuffer[:,:,:,i] = preprocessImageStack(observation,pictureSize)
				
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
			tempScore += clipReward(reward) * discountFactor**index
			index += 1
			observationBuffer[:,:,:,:-1] = observationBuffer[:,:,:,1:]
			observationBuffer[:,:,:,-1] = preprocessImageStack(observation,pictureSize)
		
		print('Episode : {}  Frame Count : {} Score : {}'.format(episode, index, tempScore))


		cumulativeScores.append(tempScore)
		numberOfFrames.append(index)

	frameMean = np.mean(numberOfFrames)
	frameStd = np.std(numberOfFrames)
	scoreMean = np.mean(cumulativeScores)
	scoreStd = np.std(cumulativeScores)
		
	print('\nMean Reward : {} - Mean Episode length : {} \nSTD Reward : {} - STD Episode length : {}'.format(scoreMean,frameMean,scoreStd,frameStd))

	return frameMean, frameStd, scoreMean, scoreStd

def runSimpleGreedyGame(env, numberOfEpisodes, maximumLength, sess, qVector, currentState):

	global real_time_render
	global render_option
	global discountFactor

	############### INITIALIZING ################## 
	episodeLength = np.zeros([numberOfEpisodes]).astype(int)
	rewards = np.zeros([numberOfEpisodes])
	################ SIMULATING ##################


	for i_episode in range(numberOfEpisodes):
		#print('EPISODE : ', i_episode)
		observation = env.reset()
		for t in range(maximumLength):
			if real_time_render:
					update_render_option()
			if render_option :
				env.render()

			# env.render()
			action = greedy_action(currentState, np.array([observation]), sess, qVector)
			observation, reward, done, info = env.step(action)
			if (done or (t + 1 == maximumLength)):
				episodeLength[i_episode] = t +1 
				if t+1 != maximumLength:
					rewards[i_episode] = -(discountFactor)**(t)
				break 
		print('episode {} -  cumulative reward : {} - episode length {}'.format(i_episode, rewards[i_episode],episodeLength[i_episode]))		
	
	print('\nMean Reward : {} - Mean Episode length : {} \nSTD Reward : {} - STD Episode length : {}'.format(np.mean(rewards),np.mean(episodeLength),np.std(rewards), np.std(episodeLength)))
	return rewards, episodeLength


################################ INITIALIZING SCRIPT #############################
listOfModels = ['3-a','3-b', '4', '5', '6', '7', '8', 'b']
listOfGames = ['CartPoleModified-v0','Pong-v3','MsPacman-v3','Boxing-v3']
listOfGamesForB = ['Pong','PacMan','Boxing']

modelFolderPath = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2]) + os.path.sep + 'models' + os.path.sep

MAX_EPISODE_LEN = 300
gym.envs.register(
    id = 'CartPoleModified-v0',
    entry_point = 'gym.envs.classic_control:CartPoleEnv',
    max_episode_steps = MAX_EPISODE_LEN,
)


print('------------- Wellcome to the testing module ------------')
print('Please select a game : ')
for question in listOfModels:
	print('for question {} --> type "{}"'.format(question,question))
questionChosen  = input('Enter the code of the model you want to test :').strip().lower()
print('You have chosen : {}'.format(questionChosen))

render_option  = 'y' in input('Would like to render the testing ? [Y|N] : ').strip().lower()


warning = input('WARNING : At any moment you can disable or enable the rendering by typing "r" (and enter if running on a UNIX system), type enter to continue ...')

isForProblemB = False

if questionChosen != 'b':
	game = listOfGames[0]
	env = gym.make(game)
	preString = ''
	postString = '0_'
	folderName = 'a{}'.format(questionChosen)
	if questionChosen == '3-a':
		folderName  = 'a3' + os.path.sep + 'part1' 
		learningRate  = float(input('choose a learning rate [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5] :').strip().lower())
		postString = ''

		currentState, qMatrix = buildModel3_1(learningRate/10)
	elif questionChosen == '3-b':
		folderName  = 'a3' + os.path.sep + 'part2' 
		learningRate  = float(input('choose a learning rate [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5] :').strip().lower())
		postString = ''
		currentState, qMatrix = buildModel3_2()
	elif questionChosen == '4':
		postString = '{}_'.format(np.random.randint(0,100))
		learningRate  = 0.01
		currentState, qMatrix = buildModel4To6(100,learningRate/10)
	elif questionChosen == '5':
		learningRate  = 0.01
		hiddenSize  = int(input('choose a hidden size [30 or 1000] :').strip().lower())
		preString = 'hiddenLayer_{}_'.format(hiddenSize)
		currentState, qMatrix = buildModel4To6(hiddenSize,learningRate/10)
	elif questionChosen == '6':
		learningRate  = 0.015
		currentState, qMatrix = buildModel4To6(100,learningRate/10)
	elif questionChosen == '7':
		learningRate  = 0.00001
		currentState, qMatrix = buildModel7(100,learningRate/10)
	elif questionChosen == '8':
		learningRate = 0.035
		currentState, qMatrix = buildModel8(100,learningRate/10)

 
else : 
	isForProblemB = True
	game = input('please choose a game {} : '.format(listOfGamesForB)).strip().lower()
	if  'pong' in  game:
		game = listOfGames[1]
	elif 'pacman' in game:
		game = listOfGames[2]
	else:
		game = listOfGames[3]

	env = gym.make(game)
	folderName = 'b3' + os.path.sep + game
	preString = ''
	postString = ''
	size =  int(input('choose a learning rate [28, 60]').strip().lower())
	if size != 60:
		size = 28
	if size == 28:
		learningRate  = float(input('choose a learning rate [0.0001, 0.001]').strip().lower())
	else:
		learningRate = 0.0001
		preString = 'size_{}_'.format(size)

	actionDim = env.action_space.n
	currentState, qMatrix =  buildModelB3(actionDim, stackSize= 4,pictureSize = size, stride = 2, std = learningRate/10)

###############

currentPath = modelFolderPath + folderName + os.path.sep
modelPath = currentPath + preString + 'Model_' + postString + str(learningRate) + '.ckpt'
numberOfEpisodes = int(input('Please enter the number of episodes for the test : '))

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	sess.run(init)
	print('LOADING : {}'.format(modelPath))
	saver.restore(sess, modelPath)
	print('LOADING SUCCESSFUL')
	if isForProblemB:
		runGreedyGame(env, sess,qMatrix, game, numberOfEpisodes, 4, size)
	else:
		runSimpleGreedyGame(env, numberOfEpisodes, 300, sess, qMatrix, currentState)