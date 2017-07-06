# Atari Games with Reinforcement Learning

*author* : **SOFIANE MAHIOU**

## Overall presentation : 

this repository aims to present the work realised in an attempt to train a set of machine learnings models to play simple atari games this Folder contains a **code/test_mode.py** script allowing you, to test all attempted models.

## Package required :	

In order to run the various scripts, make sure you have the following packages : 

* gym
* tensorflow
* numpy
* time
* Image
* matplotlib.pyplot
* sys
* platform
* time
* os
* gym[atari]

## Running 

In order to test any trained model, please **go to the root folder of this repository** within a terminal type in the following commands

```sh
cd code/
python test_model.py

```

Once this is entered, the script will first ask you to type in a **question number**. Please type one of the following : 
* A **number between 1 and 8** respectively for question 1 to 8, please read the ** Problem A **  seciton
* Or type **b** please read the ** Problem B **  seciton

Once the question number entered, it will then ask you whether or not you want  to render the games during the testing: ** type either 'y' for yes or 'n' for n ** [ keep in mind that you can always disable or renable the rendering by type r during the run (enter if you're on a UNIX machine)]


### PROBLEM A

if the question selected is not **3-a or 3b** please go to the last section.

For 3-a and 3-b, you will be asked to choose a learning rate to test, please type in one of the following learning rates **[0.00001, 0.0001, 0.001,0.01,0.1,0.5]** 


### PROBLEM B 

you will then have to choose a game, please type in one of the three : 

* **pong** for 'Pong-V3'
* **pacman** for 'MsPacman-V3'
* **boxing** for 'Boxing-v3'

you will then have to choose an image size two are available : **28 or 60**

**if you choose 28** :  you will then have to specify a learning rate of either **0.001 or 0.0001**


### FINAL SECTION : EPIOSDE NUMBER

Finally, you will have to speficiy the number of episodes to run during the testing

**RQ : Keep in mind that you can always enable and disable the render by typing r + enter** 
