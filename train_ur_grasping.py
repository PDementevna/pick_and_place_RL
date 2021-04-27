import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env, env_checker
from stable_baselines import PPO2
import ur_env

# multiprocess environment
# env = make_vec_env('CartPole-v1', n_envs=4)

env = gym.make('ur_env:ur-v0')
print(f'env : {env}')
env_checker.check_env(env, True, True)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo2_ur")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_ur")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()