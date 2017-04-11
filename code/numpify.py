import numpy as np
def numpyify(listOfLists):
	listToReturn = []
	for lst in listOfLists:
		listToReturn.append(np.array(lst))
	return listToReturn
