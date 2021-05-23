import gym
import imageio
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('ur_env:ur-v0')
env_vec = DummyVecEnv([lambda: env])
model = PPO2.load("trained_models/ppo2_trial16_10e7")


def evaluate_performance():
    numIterations = 10
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
    print(f'caught: {caught}')
    print(f'lifted: {lifted}')
    print(f'moved: {moved}')


obs = env.reset()
done = False
#
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    # print(f'reward: {rewards}')
    print(f'info: {info}')
    env.render(mode='human')


# evaluate_performance()


