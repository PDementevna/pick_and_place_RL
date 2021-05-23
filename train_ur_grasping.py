import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('ur_env:ur-v0')
env_vec = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./tb_last/')

model.learn(total_timesteps=10000000, tb_log_name="ppo2_trial16_10e7")
model.save("trained_models/ppo2_trial16_10e7")
