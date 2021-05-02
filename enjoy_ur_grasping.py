import gym
import imageio
from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv

video_folder = 'logs/videos/'
video_length = 100
from gym.wrappers import Monitor
import ur_env

# env = Monitor(gym.make('ur_env:ur-v0'), './video', force=True)
env = gym.make('ur_env:ur-v0')
env_vec = DummyVecEnv([lambda: env])
model = PPO2.load("trained_models/ppo2_ur_limited_sector")

env = VecVideoRecorder(env_vec, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="random-agent-ur")

# env.reset()

# print(f'env : {env}')
# env_checker.check_env(env, True, True)

# Enjoy trained agent
obs = env.reset()
done = False

for _ in range(video_length + 1):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='rga_array')


# while not done:
# # for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     env.render(mode='human')
