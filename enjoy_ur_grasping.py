import gym
import imageio
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import time

env = gym.make('ur_env:ur-v0')
env_vec = DummyVecEnv([lambda: env])
model = PPO2.load("trained_models/ppo2_trial16_10e7")


def evaluate_performance():
    numIterations = 1000
    caught = []
    lifted = []
    moved = []

    for i in range(numIterations):
        curCaught = False
        curLifted = False
        curMoved = False

        obs = env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render(mode='human')
            curCaught = info['isCaught']
            curLifted = info['isLifted']
            curMoved = info['isMoved']
        caught.append(curCaught)
        lifted.append(curLifted)
        moved.append(curMoved)
    caught = np.asarray(caught)
    lifted = np.asarray(lifted)
    moved = np.asarray(moved)
    print(f'caught cubes with orientation: {len(caught[caught == True])} out of {numIterations} -- {len(caught[caught == True]) / numIterations} %')
    print(f'lifted cubes with orientation: {len(lifted[lifted == True])} out of {numIterations} -- {len(lifted[lifted == True]) / numIterations} %')
    print(f'moved cubes with orientation: {len(moved[moved == True])} out of {numIterations} -- {len(moved[moved == True]) / numIterations} %')


# obs = env.reset()
# done = False
# #
# while not done:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     # print(f'reward: {rewards}')
#     print(f'info: {info}')
#     env.render(mode='human')

startTime = time.time()
evaluate_performance()
endTime = time.time()
elapsedTime = endTime - startTime
print(f'Elapsed time: {elapsedTime / 60.} minutes')


