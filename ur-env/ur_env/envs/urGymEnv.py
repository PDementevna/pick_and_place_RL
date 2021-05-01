# import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# print("current_dir=" + currentdir)
# os.sys.path.insert(0, currentdir)
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
                 renders=True,
                 isDiscrete=False,
                 maxSteps=50000):
        print("URGymEnv __init__")
        self._isDiscrete = isDiscrete
        self._timeStep = 0.001
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 3
        self._cam_yaw = 130
        self._cam_pitch = -30

        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(3, 130, -30, [0, 0, 1])
        else:
            p.connect(p.DIRECT)
        # timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "urTimings.json")
        self.seed()
        # self.outerLim = [-1.0, 1.1]
        # self.internalLim = [-0.50, 0.51]

        self.cubeXLim = [0.5, 0.7]
        self.cubeYLim = [-0.7, 0.7]

        self.trayPos = [0.640000, 0.075000, 0.63]

        # print('here reset?')
        self.reset()
        observationDim = len(self.getExtendedObservation())
        # print("observationDim")
        # print(observationDim)

        observation_high = np.array([largeValObservation] * observationDim)
        if (self._isDiscrete):
            self.action_space = spaces.Discrete(7)
        else:
            action_dim = 4
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    # def move_to_cube(self, pos):
    #     # orn = self._p.getQuaternionFromEuler([0.57, 0., 0.])
    #     jointPoses = p.calculateInverseKinematics(self._ur.urUid, self._ur.urEndEffectorIndex, pos)
    #     jointPoses = list(jointPoses)
    #     jointPoses.insert(0, 0.0)
    #     jointPoses.append(0.0)
    #     # jointPoses[self.urEndEffectorIndex] = self.endEffectorAngle
    #
    #     # for joint_index in range(self._p.getNumJoints(self._ur.gripperUid)):
    #     #     self._p.resetJointState(self._ur.urUid, joint_index, jointPoses[joint_index])
    #
    #     self._p.setJointMotorControlArray(self._ur.urUid, range(8),
    #                                       p.POSITION_CONTROL,
    #                                       jointPoses)
    #
    #     ee_state = p.getLinkState(self._ur.urUid, 6)
    #     gripper_left = p.getLinkState(self._ur.gripperUid, 4)
    #     gripper_right = p.getLinkState(self._ur.gripperUid, 6)
    #     print(f'pos received: {ee_state[0]}')
    #     print(f'pos sent: {pos}')
    #     print(f'gripper pos left: {gripper_left[0]}')
    #     print(f'gripper pos right: {gripper_right[0]}')

    # def pickObject(self, pos):
    #     self.move_to_cube(pos)
    #     self._ur.closeGripper()

    def reset(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        print("URGymEnv __reset__")
        self.terminated = 0
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(numSolverIterations=150)
        self._p.setTimeStep(self._timeStep)
        # self._p.setGravity(0, 0, -9.81)

        p.loadSDF('stadium.sdf')

        self.tableUid = p.loadURDF("models/table_custom/table.urdf", [0., 0., 0.])
        self._p.changeVisualShape(self.tableUid, 1, rgbaColor=[0.466, 0.341, 0.172, 1.0])

        # xpos = 0.55 + 0.12 * random.random()
        # ypos = 0 + 0.2 * random.random()
        # ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        # orn = p.getQuaternionFromEuler([0, 0, ang])
        # self.trayUid = p.loadURDF("tray/tray.urdf",
        #                           self.trayPos,
        #                           [0.000000, 0.000000, 1.000000, 0.000000])
        self._ur = ur.UR(timeStep=self._timeStep)
        self.ee_state = p.getLinkState(self._ur.urUid, 7)
        self.pos_orient_object = self.getRandomPosOrient(0.65)

        # coords = self.pos_orient_object[0].copy()
        # coords[2] = 1.0

        # self.cubeUid = p.loadURDF("cube_small.urdf", self.pos_orient_object[0], self.pos_orient_object[1])
        self.cubeRandomPlace()
        # self.pickObject(pos)
        # for i in range(50):
        #   self.cubeRandomPlace()
        # time.sleep(0.1)
        # self.move_to_cube(coords)

        self.catched = False

        # self.squareUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.15,
        #                             orn[0], orn[1], orn[2], orn[3])

        # p.setGravity(0, 0, -10)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def getRandomPosOrient(self, z_coord=0.70):
        # x_pos = np.random.rand() * self.outerLim[1] + self.outerLim[0]
        # y_pos = np.random.rand() * self.outerLim[1] + self.outerLim[0]

        # x_pos = 0.3
        # y_pos = 0.3

        x_pos = np.random.rand() * self.cubeXLim[1] + self.cubeXLim[0]
        y_pos = np.random.rand() * self.cubeYLim[1] + self.cubeYLim[0]

        trayBoundary = 0.25

        # if ((self.trayPos[0] + trayBoundary > x_pos > self.trayPos[0] - trayBoundary) and
        #         (self.trayPos[1] + trayBoundary > y_pos > self.trayPos[1] + trayBoundary)):
        #   self.getRandomPosOrient(outerLimitation, innerLimitation)

        # if self.internalLim[1] > x_pos > self.internalLim[0]:
        #   self.getRandomPosOrient(z_coord)
        # if self.internalLim[1] > y_pos > self.internalLim[0]:
        #   self.getRandomPosOrient(z_coord)

        orient = np.random.rand() * 3.14
        angles = p.getQuaternionFromEuler([0., 0., orient])
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
        gripperOrn = gripperState[1]
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.cubeUid)

        invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
        gripperMat = p.getMatrixFromQuaternion(gripperOrn)
        dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
        dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
        dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

        gripperEul = p.getEulerFromQuaternion(gripperOrn)
        # print("gripperEul")
        # print(gripperEul)
        cubePosInGripper, cubeOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                  cubePos, cubeOrn)
        projectedBlockPos2D = [cubePosInGripper[0], cubePosInGripper[1]]
        cubeEulerInGripper = p.getEulerFromQuaternion(cubeOrnInGripper)
        # print("projectedBlockPos2D")
        # print(projectedBlockPos2D)
        # print("cubeEulerInGripper")
        # print(cubeEulerInGripper)

        # we return the relative x,y position and euler angle of block in gripper space
        cubeInGripperPosXYEulZ = [cubePosInGripper[0], cubePosInGripper[1], cubeEulerInGripper[2]]

        # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
        # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
        # p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

        self._observation.extend(list(cubeInGripperPosXYEulZ))
        return self._observation

    def isObjectCatched(self, threshold=0.05):
        cubePos, cubeOrn = self._p.getBasePositionAndOrientation(self.cubeUid)
        gripperLeftFinger = self._p.getLinkState(self._ur.gripperUid, 3)[0]
        gripperRightFinger = self._p.getLinkState(self._ur.gripperUid, 5)[0]
        # print(f'gripper left: {gripperLeftFinger}')
        gripperEndEffPos = ((gripperRightFinger[0] + gripperLeftFinger[0]) / 2.,
                            (gripperRightFinger[1] + gripperLeftFinger[1]) / 2.,
                            (gripperRightFinger[2] + gripperLeftFinger[2]) / 2.)
        print(f'mean pos gripper: {gripperEndEffPos}')
        distance = np.sqrt((cubePos[0] - gripperEndEffPos[0]) ** 2 +
                           (cubePos[1] - gripperEndEffPos[1]) ** 2 +
                           (cubePos[2] - gripperEndEffPos[2]) ** 2)
        print(f'distance to cube: {distance}')

        if (distance < threshold):
            # print(f'distance is under threshold!')
            self.catched = True



    def gripperOpenning(self):
        if (self.catched):
            return 0.5
        return 0


    def step(self, action):
        # self.ee_state = p.getLinkState(self._ur.urUid, 7)
        # # print(f'pos 6 link: {p.getLinkState(self._ur.urUid, 6)[0]}')
        # print(f'pos 7 link: {p.getLinkState(self._ur.urUid, 7)[0]}')
        # gripperState = p.getLinkState(self._ur.gripperUid, 0)
        # gripperPos = gripperState[0]
        # # cubePos, cubeOrn = p.getBasePositionAndOrientation(self.cubeUid)
        # print(f'pos gripper base: {gripperPos}')
        # gripperLeftFinger = p.getLinkState(self._ur.gripperUid, 3)
        # gripperRightFinger = p.getLinkState(self._ur.gripperUid, 5)
        # print(f'pos gripper left: {gripperLeftFinger[0]}')
        # print(f'pos gripper right: {gripperRightFinger[0]}')
        #
        # print(f'orn ee: {self._p.getEulerFromQuaternion(self.ee_state[1])}')
        # # posEE = self.ee_state[0]
        # # action = [(self.pos_orient_object[0][0] - posEE[0]),
        # #           (self.pos_orient_object[0][1] - posEE[1]),
        # #           (self.pos_orient_object[0][2] - posEE[2]),
        # #           (self.pos_orient_object[1][2] - self.ee_state[1][2])]
        #
        # # action =
        #
        # # print(f'action is: {action}')
        if (self._isDiscrete):
            dv = 0.005
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
            f = self.gripperOpenning()
            realAction = [dx, dy, -0.002, da, f]
            # print(f'realAction right: ({realAction[0], realAction[1], realAction[2], realAction[3], realAction[4]}')
        else:
            # print("action[0]=", str(action[0]))
            dv = 0.005
            dx = action[0] * dv
            dy = action[1] * dv
            dz = action[2] * dv
            # da = 0.05
            da = action[3] * 0.05
            f = self.gripperOpenning()
            realAction = [dx, dy, dz, da, f]
            print(f'realAction else: ({realAction[0], realAction[1], realAction[2], realAction[3], realAction[4]}')
        return self.step2(realAction)

    def step2(self, action):
        for i in range(self._actionRepeat):
            self._ur.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
            print(f'env ecounter: {self._envStepCounter}')
        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()

        # print("self._envStepCounter")
        # print(self._envStepCounter)
        # self._check_accident()
        done = self._termination()

        npaction = np.array([
            # 0
            # action[0],
            # action[1],
            action[3]
        ])  # only penalize rotation until learning works well [action[0],action[1],action[3]])
        print((f'npartion: {npaction}'))
        actionCost = np.linalg.norm(npaction) * 10.
        # print("actionCost")
        # print(actionCost)
        reward = self._reward() - actionCost
        print(f'reward: {reward}')
        # print("reward")
        # print(reward)

        # print("len=%r" % len(self._observation))

        return np.array(self._observation), reward, done, {}

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
        # print (self._kuka.endEffectorPos[2])
        state = p.getLinkState(self._ur.urUid, self._ur.urEndEffectorIndex)
        actualEndEffectorPos = state[0]

        # print("self._envStepCounter")
        # print(self._envStepCounter)
        if (self.terminated or self._envStepCounter > self._maxSteps):
            print(f'envCounter, reset: {self._envStepCounter}')
            self._observation = self.getExtendedObservation()
            return True
        maxDist = 0.005
        # closestPoints = p.getClosestPoints(self.tableUid, self._ur.urUid, maxDist)

        # if (len(closestPoints)):  #(actualEndEffectorPos[2] <= -0.43):
        #   self.terminated = 1
        #
        #   # print("terminating, closing gripper, attempting grasp")
        #   # #start grasp and terminate
        #   # fingerAngle = 0.5
        #   # for i in range(100):
        #   #   graspAction = [0, 0, 0.0001, 0, fingerAngle]
        #   #   self._ur.applyAction(graspAction)
        #   #   p.stepSimulation()
        #   #   fingerAngle = fingerAngle - (0.3 / 100.)
        #   #   if (fingerAngle < 0):
        #   #     fingerAngle = 0
        #   #
        #   # for i in range(1000):
        #   #   graspAction = [0, 0, 0.001, 0, fingerAngle]
        #   #   self._ur.applyAction(graspAction)
        #   #   p.stepSimulation()
        #   #   cubePos, cubeOrn = p.getBasePositionAndOrientation(self.cubeUid)
        #   #   if (cubePos[2] > 0.90):
        #   #     #print("BLOCKPOS!")
        #   #     #print(cubePos[2])
        #   #     break
        #   #   state = p.getLinkState(self._ur.urUid, self._ur.urEndEffectorIndex)
        #   #   actualEndEffectorPos = state[0]
        #   #   if (actualEndEffectorPos[2] > 1.0):
        #   #     break
        #
        #   self._observation = self.getExtendedObservation()
        #   return True
        return False

    def _check_accident(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.cubeUid)
        if (cubePos[0] != self.pos_orient_object[0][0]
                or cubePos[1] != self.pos_orient_object[0][1]
                or cubePos[2] != self.pos_orient_object[0][2]):
            if (not self.catched):
                self.terminated = 1

    def _distance_gripper_cube(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.cubeUid)
        # joints = self._p.getNumJoints(self._ur.gripperUid)
        # print(f'num joints: {joints}')
        link_state = self._ur.getGripperPosLink(7)
        # print(f'link state: {link_state}')
        distance = math.sqrt(
            (cubePos[0] - link_state[0]) ** 2 + (cubePos[1] - link_state[1]) ** 2 + (cubePos[2] - link_state[2]) ** 2)
        # print(f'distance: {distance}')
        return distance

    def _reward(self):

        # rewards is height of target object
        cubePos, cubeOrn = self._p.getBasePositionAndOrientation(self.cubeUid)
        # print(f'pos cube reward: ({cubePos[0]}, {cubePos[1]}, {cubePos[2]})')
        closestPoints = self._p.getClosestPoints(self.cubeUid, self._ur.urUid, 2000, -1,
                                           self._ur.urEndEffectorIndex)
        distance = self._distance_gripper_cube()
        reward = -1000

        numPt = len(closestPoints)
        # print(f'num points: {numPt}')
        if (numPt > 0):
            # print("reward:")
            reward = -closestPoints[0][8] * 10
            # if (closestPoints[0][8] < 0.1):
            if (distance < 0.2):
                reward = reward + 5000

            self.isObjectCatched(0.05)

            # if (cubePos[2] > 0.75):
            if (self.catched):
                # self.catched = True
                reward = reward + 5000
                print("successfully grasped a block!!!")
                # print("self._envStepCounter")
                # print(self._envStepCounter)
                # print("self._envStepCounter")
                # print(self._envStepCounter)
                # print("reward")
                # print(reward)
            # print("reward")
            # print(reward)
            # else:
            #     self.catched = False
        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
