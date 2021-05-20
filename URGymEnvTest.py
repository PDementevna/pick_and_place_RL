import os
import inspect

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, parentdir)
from ur_env.envs import urGymEnv


def main():
	environment = urGymEnv.URGymEnv(renders=True, isDiscrete=False, maxSteps=10000000)
	motorsIds = []
	dv = 1
	motorsIds.append(environment._p.addUserDebugParameter("posX", -dv, dv, 0))
	motorsIds.append(environment._p.addUserDebugParameter("posY", -dv, dv, 0))
	motorsIds.append(environment._p.addUserDebugParameter("posZ", -dv, dv, 0))
	motorsIds.append(environment._p.addUserDebugParameter("rotation EE", -dv, dv, 0))
	motorsIds.append(environment._p.addUserDebugParameter("degreesOfClosing", 0, 100, 0))
	done = False
	while not done:
		action = []
		for motorId in motorsIds:
			action.append(environment._p.readUserDebugParameter(motorId))
		state, reward, done, info = environment.step(action)
		# environment.move_to_cube()
		obs = environment.getExtendedObservation()
		print(f'robot ee pos: {obs[0]}, {obs[1]}, {obs[2]}')


# if __name__ == "__main__":
main()
