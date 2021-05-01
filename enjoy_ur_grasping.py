import gym
from stable_baselines import PPO2
from gym.wrappers import Monitor
import ur_env
env = Monitor(gym.make('ur_env:ur-v0'), './video', force=True)
# env = gym.make('ur_env:ur-v0')
# env_vec = DummyVecEnv([lambda: env])
model = PPO2.load("trained_models/ppo2_ur_limited_sector")

print(f'env : {env}')
# env_checker.check_env(env, True, True)

# Enjoy trained agent
obs = env.reset()
done = False
while not done:
# for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='human')
