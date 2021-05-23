import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('ur_env:ur-v0')
env_vec = DummyVecEnv([lambda: env])
model = PPO2.load("trained_models/ppo2_trial16_10e7.zip")

model.set_env(env_vec)

model.learn(total_timesteps=10000000, tb_log_name="ppo2_trial16_advanced_10e7")
model.save("trained_models/ppo2_trial16_advanced_10e7")
