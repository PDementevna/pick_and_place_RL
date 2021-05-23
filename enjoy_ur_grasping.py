import gym
import imageio
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('ur_env:ur-v0')
env_vec = DummyVecEnv([lambda: env])
model = PPO2.load("trained_models/ppo2_trial21_10e7")

# Enjoy trained agent
obs = env.reset()
done = False


while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    # print(f'reward: {rewards}')
    env.render(mode='human')
