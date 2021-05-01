import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env, env_checker
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import ur_env

# multiprocess environment
# env = make_vec_env('CartPole-v1', n_envs=4)

env = gym.make('ur_env:ur-v0')
# env_vec = DummyVecEnv([lambda: env])
print(f'env : {env}')
print(f'env action space: {env.action_space}')
print(f'env observation space: {env.observation_space}')
# env_checker.check_env(env, True, True)

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./ppo2_ur_tensorboard/')
model.learn(total_timesteps=300000, tb_log_name="limited_sector")
model.save("trained_models/ppo2_ur_limited_sector")
