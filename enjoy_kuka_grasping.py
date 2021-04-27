import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env, env_checker
from stable_baselines import PPO2
import kuka_env

model = PPO2.load("ppo2_kuka")

env = gym.make('kuka_env:kuka-v0')
print(f'env : {env}')
env_checker.check_env(env, True, True)

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()