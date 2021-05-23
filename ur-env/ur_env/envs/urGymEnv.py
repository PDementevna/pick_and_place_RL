import math

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from ur_env.envs import ur
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 1000

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class URGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 # renders=False,
                 renders=True,
                 isDiscrete=False,
                 maxSteps=20000,
                 cubeOrien=False):
        print("URGymEnv __init__")
        self._isDiscrete = isDiscrete
        self._timeStep = 0.01
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.cubeOrn = cubeOrien
        self.terminated = 0
        self._cam_dist = 3
        self._cam_yaw = 130
        self._cam_pitch = -30
        self.info = {
            'isCaught': False,
            'isLifted': False,
            'isMoved': False
        }
        self.placePoint = [0.64, -0.63, 1.0]

        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(3, 130, -30, [0, 0, 1])
        else:
            p.connect(p.DIRECT)
        self.seed()

        # self.cubeXLim = [0.4, 0.5]
        self.cubeXLim = [0.4, 0.7]
        # self.cubeYLim = [0.0, 0.1]
        self.cubeYLim = [-0.2, 0.3]

        self.trayPos = [0.640000, -0.6, 0.63]

        self.reset()
        observationDim = len(self.getExtendedObservation())

        observation_high = np.array([largeValObservation] * observationDim)
        if (self._isDiscrete):
            self.action_space = spaces.Discrete(7)
        else:
            action_dim = 5
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    def _setInfo(self, caught=False, lifted=False, moved=False):
        self.info['isCaught'] = caught
        self.info['isLifted'] = lifted
        self.info['isMoved'] = moved


    def reset(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        print("URGymEnv __reset__")
        self.terminated = 0
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(numSolverIterations=150)
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -9.81)

        p.loadSDF('stadium.sdf')

        self.tableUid = p.loadURDF("models/table_custom/table.urdf", [0., 0., 0.])
        self._p.changeVisualShape(self.tableUid, 1, rgbaColor=[0.466, 0.341, 0.172, 1.0])

        self.trayUid = p.loadURDF("models/tray/tray.urdf",
                                  self.trayPos,
                                  [0.000000, 0.000000, 1.000000, 0.000000])
        self._p.changeVisualShape(self.trayUid, 1, rgbaColor=[0.124, 0.090, 0.083, 1.0])

        self._ur = ur.UR(timeStep=self._timeStep)
        self.ee_state = p.getLinkState(self._ur.urUid, 6)
        self.pos_orient_object = self.getRandomPosOrient(0.65)


        # CREATING THE CUBE
        self.cubeRandomPlace()
        self._p.changeVisualShape(self.cubeUid, 1, rgbaColor=[0.022, 0.223, 0.026, 1.0])

        # p.loadURDF("cube_small.urdf", [self.cubeXLim[0], self.cubeYLim[0], 0.65])
        # p.loadURDF("cube_small.urdf", [self.cubeXLim[0], self.cubeYLim[1], 0.65])
        # p.loadURDF("cube_small.urdf", [self.cubeXLim[1], self.cubeYLim[0], 0.65])
        # p.loadURDF("cube_small.urdf", [self.cubeXLim[1], self.cubeYLim[1], 0.65])

        self.catched = False
        self.lifted = False
        self.moved = False

        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def getRandomPosOrient(self, z_coord=0.70):

        baseX = np.random.rand() * (self.cubeXLim[1] - self.cubeXLim[0])
        baseY = np.random.rand() * (self.cubeYLim[1] - self.cubeYLim[0])
        # print(f'base x: {baseX}, base Y: {baseY}')

        x_pos = baseX + self.cubeXLim[0]
        y_pos = baseY + self.cubeYLim[0]
        # x_pos = 0.45
        # y_pos = 0.1

        if (self.cubeOrn):
            orient = np.random.rand() * (math.pi / 2.0)
        else:
            orient = 0.0
        angles = p.getQuaternionFromEuler([0., 0., orient])
        # print(f'x cube pos: {x_pos}, y cube pos: {y_pos}')
        return [[x_pos, y_pos, z_coord], angles]

    def cubeRandomPlace(self):
        pos_orn = self.pos_orient_object
        self.cubeUid = p.loadURDF("cube_small.urdf", pos_orn[0], pos_orn[1])
        return pos_orn[0], pos_orn[1]

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        self._observation = self._ur.getObservation()
        gripperState = p.getLinkState(self._ur.gripperUid, 0)
        gripperPos = gripperState[0]
        # print()
        # print(f'gripper base pos: {gripperPos}')
        gripperOrn = gripperState[1]
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.cubeUid)
        # print(f'cube base pos: {cubePos}')

        cubePos, cubeOrn = self._p.getBasePositionAndOrientation(self.cubeUid)
        gripperLeftFinger = self._p.getLinkState(self._ur.gripperUid, 3)[0]
        gripperRightFinger = self._p.getLinkState(self._ur.gripperUid, 5)[0]
        # print(f'gripper left: {gripperLeftFinger}')
        gripperEndEffPos = [(gripperRightFinger[0] + gripperLeftFinger[0]) / 2.,
                            (gripperRightFinger[1] + gripperLeftFinger[1]) / 2.,
                            (gripperRightFinger[2] + gripperLeftFinger[2]) / 2.]

        # substitute the distance on z axis from center of mass to the end-effector point
        gripperEndEffPos[2] -= 0.038
        gripperEndEffPos = tuple(gripperEndEffPos)

        # print(f'gripper EE position {gripperEndEffPos}')



        invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
        invGripperPosEE, invGripperOrnEE = p.invertTransform(gripperEndEffPos, gripperOrn)
        gripperMat = p.getMatrixFromQuaternion(gripperOrn)
        dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
        dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
        dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

        gripperEul = p.getEulerFromQuaternion(gripperOrn)
        # print("gripperEul")
        # print(gripperEul)
        cubePosInGripper, cubeOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                  cubePos, cubeOrn)

        cubePosInGripperEE, cubeOrnInGripperEE = p.multiplyTransforms(invGripperPosEE, invGripperOrnEE,
                                                                  cubePos, cubeOrn)

        # print(f'cube pos in gripper: {cubePosInGripper}')
        # print(f'cube pos in gripper EE: {cubePosInGripperEE}')
        projectedBlockPos2D = [cubePosInGripper[0], cubePosInGripper[1]]
        projectedBlockPosEE2D = [cubePosInGripperEE[0], cubePosInGripperEE[1]]
        cubeEulerInGripper = p.getEulerFromQuaternion(cubeOrnInGripper)
        # print("projectedBlockPos2D")
        # print(projectedBlockPos2D)
        # print("cubeEulerInGripper")
        # print(cubeEulerInGripper)

        # we return the relative x,y position and euler angle of block in gripper space
        cubeInGripperPosXYEulZ = [cubePosInGripper[0], cubePosInGripper[1], cubeEulerInGripper[2]]
        cubeInGripperPosEEXYEulZ = [cubePosInGripperEE[0], cubePosInGripperEE[1], cubeEulerInGripper[2]]

        # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
        # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
        # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

        # self._observation.extend(list(cubeInGripperPosXYEulZ))
        self._observation.extend(list(cubeInGripperPosEEXYEulZ))

        return self._observation

    def gripperOpenning(self):
        if (self.catched):
            return 0.5
        return 0


    def step(self, action):
        if (self._isDiscrete):
            dv = 0.005
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
            f = self.gripperOpenning()
            # realAction = [dx, dy, -0.002, da, f]
            # print(f'realAction right: ({realAction[0], realAction[1], realAction[2], realAction[3], realAction[4]}')
        else:
            dv = 0.003
            dx = action[0] * dv
            dy = action[1] * dv
            dz = action[2] * dv
            # da = 0.05
            da = action[3] * 0.005
            # da = 0.0
            # f = self.gripperOpenning()
            gripperState = 0.0
            # gripperState = action[4] * 0.008
            # realAction = [dx, dy, -0.005, da, gripperState]
            realAction = [dx, dy, -0.0002, da, gripperState]
            # realAction = [dx, dy, -0.00005, da, gripperState]
            # realAction = [dx, dy, dz, da, gripperState]
            # print(f'realAction else: ({realAction[0], realAction[1], realAction[2], realAction[3], realAction[4]}')
        return self.step2(realAction)

    def step2(self, action):
        for i in range(self._actionRepeat):
            self._ur.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
            # print(f'env ecounter: {self._envStepCounter}')
        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()

        # print("self._envStepCounter")
        # print(self._envStepCounter)
        # self._check_accident()
        done = self._termination()
        # done = False

        npaction = np.array([
            # action[0],
            # action[1],
            action[3]
        ])  # only penalize rotation until learning works well [action[0],action[1],action[3]])
        actionCost = np.linalg.norm(npaction) * 10.
        reward = self._reward() - actionCost
        # print(f'reward: {reward}')
        # print("reward")
        # print(reward)

        # print("len=%r" % len(self._observation))

        return np.array(self._observation), reward, done, self.info

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        # self.reset()
        cube_pos, orn_cube = self._p.getBasePositionAndOrientation(self.cubeUid)
        # print(f'cube pos: {cube_pos}; cube orn: {orn_cube}')
        base_pos, orn = self._p.getBasePositionAndOrientation(self._ur.urUid)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=4)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                  height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        # print('termination process')
        # print (self._kuka.endEffectorPos[2])
        state = p.getLinkState(self._ur.urUid, self._ur.urEndEffectorIndex)
        distance = self._distance_gripper_cube(self.cubeUid)
        actualEndEffectorPos = state[0]

        # print("self._envStepCounter")
        # print(self._envStepCounter)
        if (self.terminated or self._envStepCounter > self._maxSteps):
            # print(f'envCounter, reset: {self._envStepCounter}')
            self._observation = self.getExtendedObservation()
            # print('termination')
            return True

        maxDist = 0.0009
        closestPoints = p.getClosestPoints(self.cubeUid, self._ur.gripperUid, maxDist)
        # print(f'len closest points: {len(closestPoints)}')

        isMoved = False
        numIter = 1000

        if (distance <= 0.01005):
        # if (len(closestPoints)):
            self.terminated = 1

            print("terminating, closing gripper, attempting grasp")
            degreeOfGripper = 0.0

            # close gripper
            for i in range(100):
                print('gripper is closing')
                graspAction = [0., 0., -0.00002, 0., degreeOfGripper]
                self._ur.applyAction(graspAction)
                p.stepSimulation()
                degreeOfGripper += 0.006
                if (degreeOfGripper > 0.6):
                    degreeOfGripper = 0.6

                contactPointsLeft = p.getContactPoints(self._ur.gripperUid, self.cubeUid, 3, -1)
                contactPointsRight = p.getContactPoints(self._ur.gripperUid, self.cubeUid, 5, -1)
                distance = self._distance_gripper_cube(self.cubeUid)
                # if (len(contactPointsRight) > 0 or len(contactPointsLeft) > 0 or distance < 0.009):
                if (len(contactPointsRight) > 0 and len(contactPointsLeft) > 0):
                    print('the contact is established!')
                    # print(f'distance of contact is -- left: {contactPointsLeft[0][8]} -- right: {contactPointsRight[0][8]}')
                    self.catched = True
                    # break
                # else:
                #     self.catched = False


            # lift the cube
            for i in range(1000):
                print('gripper lifting')
                graspAction = [0., 0., 0.0005, 0, degreeOfGripper]

                degreeOfGripper += 0.0001
                if (degreeOfGripper > 0.7):
                    degreeOfGripper = 0.7

                self._ur.applyAction(graspAction)
                p.stepSimulation()
                cubePos, cubeOrn = p.getBasePositionAndOrientation(self.cubeUid)
                if (cubePos[2] > 0.76):
                    print('aaaaaa it is lifted!!!!')
                    self.lifted = True
                    break
                state = self._p.getLinkState(self._ur.gripperUid, 1)
                actualGripperPos = state[0]
                if (actualGripperPos[2] > 0.85):
                    break


            currPos = p.getLinkState(self._ur.urUid, 6)[0]
            xStep = (self.placePoint[0] - currPos[0]) / numIter
            yStep = (self.placePoint[1] - currPos[1]) / numIter
            zStep = (self.placePoint[2] - currPos[2]) / numIter

            for i in range(numIter):
                print('gripper moving')

                moveAction = [xStep, yStep, zStep, 0, degreeOfGripper]
                self._ur.applyAction(moveAction)
                p.stepSimulation()

            for i in range(50):
                print('gripper is opening')
                contactPoints = p.getClosestPoints(self.trayUid, self.cubeUid, 0.01)
                openAction = [0., 0., 0., 0., -0.005]
                self._ur.applyAction(openAction)
                p.stepSimulation()
                if (len(contactPoints) > 0):
                    print('the cube is place successfully')
                    self.moved = True
                    break

            self._observation = self.getExtendedObservation()
            self._setInfo(self.catched, self.lifted, self.moved)
            return True
        return False


    def _distance_gripper_cube(self, cubeID):

        cubePos, cubeOrn = self._p.getBasePositionAndOrientation(cubeID)
        gripperLeftFinger = self._p.getLinkState(self._ur.gripperUid, 3)[0]
        gripperRightFinger = self._p.getLinkState(self._ur.gripperUid, 5)[0]
        # print(f'gripper left: {gripperLeftFinger}')
        gripperEndEffPos = [(gripperRightFinger[0] + gripperLeftFinger[0]) / 2.,
                            (gripperRightFinger[1] + gripperLeftFinger[1]) / 2.,
                            (gripperRightFinger[2] + gripperLeftFinger[2]) / 2.]
        gripperEndEffPos[2] -= 0.038
        gripperEndEffPos = tuple(gripperEndEffPos)
        # print(f'mean pos gripper: {gripperEndEffPos}')
        distance = np.sqrt((cubePos[0] - gripperEndEffPos[0]) ** 2 +
                           (cubePos[1] - gripperEndEffPos[1]) ** 2 +
                           (cubePos[2] - gripperEndEffPos[2]) ** 2)
        return distance

    def _reward(self):

        # rewards is height of target object
        cubePos, cubeOrn = self._p.getBasePositionAndOrientation(self.cubeUid)
        # print()
        # print(f'pos cube: ({cubePos[0]}, {cubePos[1]}, {cubePos[2]})')
        closestPoints = self._p.getClosestPoints(self.cubeUid, self._ur.gripperUid, 0.075, -1, 3)
        distance = self._distance_gripper_cube(self.cubeUid)
        reward = -1000
        # print(f'distance to cube: {distance}')

        numPt = len(closestPoints)
        # print(f'num points: {numPt}')
        if (numPt > 0):
            # print("reward:")
            reward = -distance * 1000
            # reward = -closestPoints[0][8] * 10

            # print(f'dist from closest points: {closestPoints[0][8]}')
            #
            # if (cubePos[2] > 0.75):
            #     reward += 10000
            #     print('lifted the cube!!!')

            # if (distance <= 0.0105):
            #     reward = reward + 10000
            #     print('the cube is close!')
            #     self.closeToCube = True
            #     self._termination()

            # self.isObjectCatched(0.035)

            # if (cubePos[2] > 0.75):
            #     if (self.catched):
            #         # self.catched = True
            #         # reward = reward + 7000
            #         print("successfully grasped a block!!!")

            if (cubePos[2] > 0.75):
                reward += 10000
                # print('the cube is close!')
                print("successfully grasped a block!!!")



        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
