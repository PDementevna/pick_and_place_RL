from gym.envs.registration import register

register(
    id='ur-v0',
    entry_point='ur_env.envs:URGymEnv',
)
# register(
#     id='foo-extrahard-v0',
#     entry_point='gym_foo.envs:FooExtraHardEnv',
# )