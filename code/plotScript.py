import os
import numpy as np
import matplotlib.pyplot as plt


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



def plotQuestion3():
	for i in range(1,3):
		numpyFilePath = 'Q3' + os.sep  + 'Part'+ str(i) + os.sep + 'Numpy' + os.sep
		saveFilePath  = 'report' + os.sep + 'plots' + os.sep + 'a3-'+ str(i) + os.sep
		learningRates = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 0.5]
		listOfPlots = ['bellmanLoss', 'loss',  'reward','step']
		if not(os.path.exists(saveFilePath)):
			os.makedirs(saveFilePath)

		for plotName in listOfPlots:
			print('Q3-PART{} : {}-plot'.format(i,plotName))
			fig = plt.figure()
			legends = []
			maxValue = -10^6
			minValue = 10^6
			for learningRate in learningRates:
				plot_array = np.load(numpyFilePath + plotName + 'Array_{}'.format(learningRate) + '.npy')
				plt.plot(plot_array, label = str(learningRate))
				
				maxValue = max(np.max(plot_array),maxValue)
				minValue = min(np.min(plot_array),minValue)

			if plotName == 'bellmanLoss' and i == 1:
				maxValue =min(maxValue,-0.03)
			elif plotName == 'loss' and i == 1:
				maxValue =min(maxValue,0.025)

			if maxValue > 0: 
				maxValue *= 1.1
			elif maxValue < 0:
				maxValue *= 0.9
			else:
				maxValue = abs(minValue)/100

			if minValue >= 0:
				minValue *= 0.9
			else : 
				minValue *= 1.1

			plt.legend(loc='lower right', fancybox = True, framealpha = 0.5)
			plt.xlabel('number of epochs')
			plt.ylabel(plotName)
			plt.axis([0, plot_array.shape[0], minValue, maxValue])
			plt.savefig(saveFilePath + plotName + '.png')
			plt.close(fig)


def plotQuestion4():
	#### parameters of the question
	questionNumber = 4
	iterationsToAvoid = range(30,100)
	numberOfIterations = 100
	learningRate = 0.01
	discount = 0.99
	#### paths and arrays
	numpyFilePath = 'Q{}'.format(questionNumber) + os.sep + 'Numpy' + os.sep
	saveFilePath  = 'report' + os.sep + 'plots' + os.sep + 'a{}'.format(questionNumber) + os.sep
	listOfPlots = ['bellmanLoss', 'loss',  'reward', 'train_step','test_step']
	iterations = [i for i in range(numberOfIterations) if i not in iterationsToAvoid]

	if not(os.path.exists(saveFilePath)):
		os.makedirs(saveFilePath)

	### STARTING THE PLOTS
	for i in range(len(listOfPlots)):
		fig = plt.figure()
		plotName = listOfPlots[i]
		print('Q{} : {}-plot'.format(questionNumber,plotName))
		plot_array = np.load(numpyFilePath + plotName + 'Array_{}'.format(learningRate) + '.npy')[iterations]
		max_Value = np.max(plot_array)
		print('Q{} : {}-plot, maxValue : {}'.format(questionNumber,plotName,max_Value))
		plot_mean = plot_array.mean(axis = 0)
		plot_std = plot_array.std(axis = 0)
		plt.plot(plot_mean)
		plt.fill_between(range(len(plot_mean)),plot_mean+plot_std,plot_mean-plot_std, facecolor = 'green', edgecolor = None,lw=0, alpha = 0.5)
		plt.xlabel('number of episodes')
		plt.ylabel(plotName)
		plt.savefig(saveFilePath + plotName + '.png')
		plt.close(fig)

	plotName = 'test_reward'
	print('Q{} : {}-plot'.format(questionNumber,plotName))
	fig = plt.figure()
	arrayString = 'test_step'
	initial_plot_array = np.load(numpyFilePath + arrayString + 'Array_{}'.format(learningRate) + '.npy')[iterations]
	plot_array = -1 * np.power(discount,initial_plot_array)
	plot_array[initial_plot_array == 300] = 0
	plot_mean = plot_array.mean(axis = 0)
	plot_std = plot_array.std(axis = 0)
	plt.plot(plot_mean, label = str(learningRate))
	plt.fill_between(range(len(plot_mean)),plot_mean+plot_std,plot_mean-plot_std, facecolor = 'green', edgecolor = None,lw=0, alpha = 0.5)
	plt.xlabel('number of episodes')
	plt.ylabel(plotName)
	plt.axis([0, plot_array.shape[1], -1, np.max(plot_array)*1.1])

	plt.savefig(saveFilePath + plotName + '.png')
	plt.close(fig)






def plotQuestion5():
	#### parameters of the question
	questionNumber = 5
	iterationsToAvoid = range(30,100)
	numberOfIterations = 1
	secondNumberOfIterations = 100
	learningRate = 0.01
	discount = 0.99
	hiddenLayers = [30,1000]
	multiplePlotsName = ['30', '1000', '100']

	#### paths and arrays
	numpyFilePaths = ['Q{}'.format(questionNumber) + os.sep + 'Numpy' + os.sep + 'hiddenLayer_{}_'.format(hl) for hl in  hiddenLayers] + ['Q{}'.format(4) + os.sep + 'Numpy' + os.sep ]
	saveFilePath  = 'report' + os.sep + 'plots' + os.sep + 'a{}'.format(questionNumber) + os.sep
	listOfPlots = ['bellmanLoss', 'loss',  'reward', 'train_step','test_step']
	iterations = [i for i in range(numberOfIterations) if i not in iterationsToAvoid]
	fullIterations = [i for i in range(secondNumberOfIterations) if i not in iterationsToAvoid]


	if not(os.path.exists(saveFilePath)):
		os.makedirs(saveFilePath)

	### STARTING THE PLOTS
	for i in range(len(listOfPlots)):
		fig = plt.figure()
		plotName = listOfPlots[i]
		maxValue = 0.01
		minValue = 0
		print('Q{} : {}-plot'.format(questionNumber,plotName))

		for j in range(len(numpyFilePaths)):
			numpyFilePath = numpyFilePaths[j]
			labelPlot = '{} layers'.format(multiplePlotsName[j])
			if j != 2:
				plot_array = np.load(numpyFilePath + plotName + 'Array_{}'.format(learningRate) + '.npy')[iterations,:]
			else:
				plot_array = np.load(numpyFilePath + plotName + 'Array_{}'.format(learningRate) + '.npy')[fullIterations,:]
			
			maxValue = max(maxValue,np.max(plot_array))
			minValue = min(minValue, np.min(plot_array))
			if 'test' in plotName and j == 2: 
				indexes = np.linspace(0, 1999, num=100).astype(int)
			else:
				indexes = np.array(range(plot_array.shape[1])).astype(int)		

			plot_mean = plot_array[:,indexes].mean(axis = 0)
			plt.plot(plot_mean, label = labelPlot)
#		plt.fill_between(range(len(plot_mean)),plot_mean+plot_std,plot_mean-plot_std, facecolor = 'green', edgecolor = None,lw=0, alpha = 0.5)
		
		plt.legend(loc='lower right', fancybox = True, framealpha = 0.5)
		plt.xlabel('number of episodes')
		plt.ylabel(plotName)
		plt.axis([0, plot_mean.shape[0],minValue , maxValue*1.1])
		plt.savefig(saveFilePath + plotName + '.png')
		plt.close(fig)


	plotName = 'test_reward'
	print('Q{} : {}-plot'.format(questionNumber,plotName))
	fig = plt.figure()
	arrayString = 'test_step'
	maxValue = 0.01
	minValue = 0
	for j in range(len(numpyFilePaths)):
		numpyFilePath = numpyFilePaths[j]
		labelPlot = '{} layers'.format(multiplePlotsName[j])
		if j != 2:
			initial_plot_array = np.load(numpyFilePath + arrayString + 'Array_{}'.format(learningRate) + '.npy')[iterations]
		else:
			initial_plot_array = np.load(numpyFilePath + arrayString + 'Array_{}'.format(learningRate) + '.npy')[fullIterations,:]
		
		if j == 2: 
			indexes = np.linspace(0, 1999, num=100).astype(int)
		else:
			indexes = np.array(range(initial_plot_array.shape[1])).astype(int)
		plot_array = -1 * np.power(discount,initial_plot_array)
		plot_array[initial_plot_array == 300] = 0
		maxValue = max(maxValue,np.max(plot_array))
		minValue = min(minValue, np.min(plot_array))
		plot_mean = plot_array[:,indexes].mean(axis = 0)
		plt.plot(plot_mean, label = labelPlot)
	plt.legend(loc='lower right', fancybox = True, framealpha = 0.5)
	plt.xlabel('number of episodes')
	plt.ylabel(plotName)
	
	plt.axis([0, plot_mean.shape[0],minValue , maxValue*1.1])

	plt.savefig(saveFilePath + plotName + '.png')
	plt.close(fig)




def plotQuestionAndCompare(questionNumber,listToCompareTo, multiplePlotsName, learningRates, iterationsToAvoid = range(30,100), numberOfIterations = 1, secondNumberOfIterations = 100,discount=0.99):
	#### parameters of the question


	#### paths and arrays
	numpyFilePaths = ['Q{}'.format(questionNumber) + os.sep + 'Numpy' + os.sep] + ['Q{}'.format(number) + os.sep + 'Numpy' + os.sep for number in listToCompareTo]
	saveFilePath  = 'report' + os.sep + 'plots' + os.sep + 'a{}'.format(questionNumber) + os.sep
	listOfPlots = ['bellmanLoss', 'loss',  'reward', 'train_step','test_step']
	iterations = [i for i in range(numberOfIterations) if i not in iterationsToAvoid]
	fullIterations = [i for i in range(secondNumberOfIterations) if i not in iterationsToAvoid]


	if not(os.path.exists(saveFilePath)):
		os.makedirs(saveFilePath)

	### STARTING THE PLOTS
	for i in range(len(listOfPlots)):
		fig = plt.figure()
		plotName = listOfPlots[i]
		maxValue = 0.01
		minValue = 0
		print('Q{} : {}-plot'.format(questionNumber,plotName))

		for j in range(len(numpyFilePaths)):
			learningRate = learningRates[j]
			numpyFilePath = numpyFilePaths[j]
			labelPlot = '{}'.format(multiplePlotsName[j])
			if j != len(listToCompareTo) or listToCompareTo[-1] != 4:
				plot_array = np.load(numpyFilePath + plotName + 'Array_{}'.format(learningRate) + '.npy')[iterations,:]
			else:
				plot_array = np.load(numpyFilePath + plotName + 'Array_{}'.format(learningRate) + '.npy')[fullIterations,:]
			
			maxValue = max(maxValue,np.max(plot_array))
			minValue = min(minValue, np.min(plot_array))
			if 'test' in plotName and j ==  len(listToCompareTo) and listToCompareTo[-1] == 4: 
				indexes = np.linspace(0, 1999, num=100).astype(int)
			else:
				indexes = np.array(range(plot_array.shape[1])).astype(int)		

			plot_mean = plot_array[:,indexes].mean(axis = 0)
			plt.plot(plot_mean, label = labelPlot)
#		plt.fill_between(range(len(plot_mean)),plot_mean+plot_std,plot_mean-plot_std, facecolor = 'green', edgecolor = None,lw=0, alpha = 0.5)
		
		plt.legend(loc='lower right', fancybox = True, framealpha = 0.5)
		plt.xlabel('number of episodes')
		plt.ylabel(plotName)
		plt.axis([0, plot_mean.shape[0],minValue , maxValue*1.1])
		plt.savefig(saveFilePath + plotName + '.png')
		plt.close(fig)


	plotName = 'test_reward'
	print('Q{} : {}-plot'.format(questionNumber,plotName))
	fig = plt.figure()
	arrayString = 'test_step'
	maxValue = 0.01
	minValue = 0
	for j in range(len(numpyFilePaths)):
		learningRate = learningRates[j]
		numpyFilePath = numpyFilePaths[j]
		labelPlot = '{}'.format(multiplePlotsName[j])
		if j != len(listToCompareTo) or listToCompareTo[-1] != 4:
			initial_plot_array = np.load(numpyFilePath + arrayString + 'Array_{}'.format(learningRate) + '.npy')[iterations]
		else:
			initial_plot_array = np.load(numpyFilePath + arrayString + 'Array_{}'.format(learningRate) + '.npy')[fullIterations,:]
		
		if j == len(listToCompareTo) and listToCompareTo[-1] == 4: 
			indexes = np.linspace(0, 1999, num=100).astype(int)
		else:
			indexes = np.array(range(initial_plot_array.shape[1])).astype(int)
		plot_array = -1 * np.power(discount,initial_plot_array)
		plot_array[initial_plot_array == 300] = 0
		maxValue = max(maxValue,np.max(plot_array))
		minValue = min(minValue, np.min(plot_array))
		plot_mean = plot_array[:,indexes].mean(axis = 0)
		plt.plot(plot_mean, label = labelPlot)
	plt.legend(loc='lower right', fancybox = True, framealpha = 0.5)
	plt.xlabel('number of episodes')
	plt.ylabel(plotName)
	
	plt.axis([0, plot_mean.shape[0],minValue , maxValue*1.1])

	plt.savefig(saveFilePath + plotName + '.png')
	plt.close(fig)


plotQuestion3()
plotQuestion4()
plotQuestion5()
plotQuestionAndCompare(6,[4], ['Q6','Q4'], [0.015,0.01])
plotQuestionAndCompare(7,[6], ['Q7','Q6'], [0.00001, 0.015])
plotQuestionAndCompare(8,[4], ['Q8','Q4'], [0.035, 0.01])



def plotQuestionB3(coupleOfParams = [(0.0001,28)]):
	#### parameters of the question
	gameList = ['Boxing-v3',"MsPacman-v3", "Pong-v3"]
	numpyFilePathToFormat = 'Problem-B' + os.sep + 'Q3' + os.sep + '{}' + os.sep + 'Numpy' + os.sep
	saveFilePathToFormat  = 'report' + os.sep + 'plots' + os.sep + 'b{}'.format(3) + os.sep + '{}' + os.sep 
	listOfPlots = ['bellmanLossArray', 'lossArray',  'numberOfFrames', 'testFrameMean','testScoreMean','trainScore']
	listOflabels = ['bellman loss', 'loss', 'training - number of frames', 'testing - number of frames', 'testing - averaged cumulative score ', "training - cumulative score"]
	learningRate = 0.0001
	averageValues = [-0.24,2.56,-0.84]

	### STARTING THE PLOTS

	for i in range(len(gameList)):
		game = gameList[i]
		numpyFilePath = numpyFilePathToFormat.format(game)
		saveFilePath = saveFilePathToFormat.format(game)

		if not(os.path.exists(saveFilePath)):
			os.makedirs(saveFilePath)

		for j in range(len(listOfPlots)):
			fig = plt.figure()
			maxLength = 0
			maxValue = -10**6
			minValue = 10**6
			plotName = listOfPlots[j]
			print('{} : {}-plot'.format(game,plotName))
			ylabel =  listOflabels[j]
			for learningRate, size in coupleOfParams:
				plotlabel = 'size : {} lr : {}'.format(size,learningRate)
				stringSize = ''
				if size != 28:
					stringSize = 'size_{}_'.format(size)
				plot_array = np.load(numpyFilePath + stringSize + plotName + '_{}'.format(learningRate) + '.npy')
				if plot_array.shape[0] != 10**4:
					maxLength = max(maxLength,plot_array.shape[0])
				else:
					plot_array = plot_array[:maxLength]
					lastValue = plot_array[plot_array != 0][-1]
					plot_array[plot_array == 0] = lastValue

				plt.plot(plot_array, label = plotlabel)
				print('learningRate: {}, Last Value : {}'.format(learningRate, plot_array[-1]))

				maxValue = max(maxValue,np.max(plot_array))
				minValue = min(minValue,np.min(plot_array))

				# if 'frame' in plotName.lower():
				# 	print(np.max(plot_array), np.min(plot_array))
				# 	print('minValue : {}, maxValue : {}'.format(minValue,maxValue))

			
			if 'score' in plotName.lower():
				plt.plot([averageValues[i]]*plot_array.shape[0], label = 'random policy')
				maxValue = max(maxValue,averageValues[i])
				minValue = min(minValue,averageValues[i])


			if maxValue > 0: 
				maxValue *= 1.01
			elif maxValue < 0:
				maxValue *= 0.99
			else:
				maxValue = abs(minValue)/100

			if minValue >= 0:
				minValue *= 0.99
			else : 
				minValue *= 1.01
			if len(coupleOfParams)>1:
				plt.legend(loc='lower right', fancybox = True, framealpha = 0.5)

			plt.xlabel('number of episodes')
			plt.ylabel(ylabel)			
			plt.axis([0, plot_array.shape[0],minValue , maxValue])
			plt.savefig(saveFilePath + plotName + '.png')
			plt.close(fig)



plotQuestionB3([(0.0001,28),(0.001,28),(0.0001,60)])