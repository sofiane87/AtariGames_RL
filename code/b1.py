import gym
import numpy as np
import sys 

gameName = ['Pong-v3','MsPacman-v3','Boxing-v3']
gameIndex = 1
argList = sys.argv[1:]
if ('-game' in argList):
	gameIndex = int(argList[argList.index('-game')+1])
game = gameName[gameIndex]

discount = 0.99
numberOfEpisodes = 100


def clipReward(reward):
	if reward <= -1:
		return -1
	elif reward >= 1:
		return 1
	else :
		return 0

def runGame(game, numberOfEpisodes,discount):
	env = gym.make(game)
	actionSpace = env.action_space.n
	print('Action Space Dimension : {}'.format(actionSpace))
	numberOfFrames = []
	cumulativeScores = []
	for episode in range(numberOfEpisodes):
		Done = False
		observation = env.reset()	
		index = 0
		tempScore = 0
		while not(Done):
			action = env.action_space.sample()
			Observation, reward, Done, info = env.step(action)
			tempScore += clipReward(reward) * (discount**index)
			index += 1
		cumulativeScores.append(tempScore)
		#numberOfFrames.append(index//4)
		numberOfFrames.append(index)
		if episode == numberOfEpisodes - 1 :
			print('episode {} -  {} frames - cumulative reward {} '.format(episode, index , tempScore))
		else:
			print('episode {} -  {} frames - cumulative reward {}  '.format(episode, index, tempScore), end = '\r')

	frameMean = np.mean(numberOfFrames)
	frameStd = np.std(numberOfFrames)
	scoreMean = np.mean(cumulativeScores)
	scoreStd = np.std(cumulativeScores)
	
	print('Mean Frame Count : {} -  Frame Count STD {}'.format(frameMean,frameStd) )
	print('Mean Score : {} -  Score STD {}'.format(scoreMean,scoreStd) )

	return frameMean, frameMean, scoreMean, scoreMean

print('Game : '+game)
runGame(game, numberOfEpisodes,discount)
