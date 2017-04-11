from gym import envs
import gym
import numpy as np

MAX_EPISODE_LEN = 300
gym.envs.register(
    id = 'CartPoleModified-v0',
    entry_point = 'gym.envs.classic_control:CartPoleEnv',
    max_episode_steps = MAX_EPISODE_LEN,
)

seed = 42 
discountFactor = 0.99
np.random.seed(seed)
print('SEED: ', seed)

print('Question 2')

env = gym.make('CartPole-v0')
numberOfEpisodes = 100
MaximumLength = 300

returnReward = np.zeros([numberOfEpisodes])
finalStep = np.zeros([numberOfEpisodes]).astype(int)

for i_episode in range(numberOfEpisodes):
    observation = env.reset()
    for t in range(MaximumLength):
        # env.render()
        action = np.random.random_integers(0,1)
        observation, reward, done, info = env.step(action)
        if (done and (t + 1 != MaximumLength)):
            returnReward[i_episode] += discountFactor**(t+1) * -1
            #print("Episode "+ str(i_episode)+" finished after {} timesteps".format(t+1))
            #print("Reward : "+ str(returnReward[i_episode]))
            finalStep[i_episode] = t+1
            break


print('Mean TimeStep: ',np.mean(finalStep))
print('Variance TimeStep: ', np.var(finalStep))

print('Mean reward: ',np.mean(returnReward))
print('Variance reward: ', np.var(returnReward))