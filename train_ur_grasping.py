import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env, env_checker
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import ur_env

# multiprocess environment

env = gym.make('ur_env:ur-v0')
env_vec = DummyVecEnv([lambda: env])
print(f'env : {env}')
# env_checker.check_env(env, True, True)

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./ppo2_ur_tensorboard/')
model.learn(total_timesteps=5000000, tb_log_name="limited_sector")
model.save("trained_models/ppo2_ur_limited_sector_small")
