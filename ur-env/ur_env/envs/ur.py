# import os, inspect
#
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data


class UR:

    def __init__(self, urdfRootPath='models/', timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = .35
        self.maxForce = 200.
        self.fingerForce = 10
        # self.fingerBForce = 2.5
        self.fingerTipForce = 2
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 21
        self.useOrientation = 1
        self.urEndEffectorIndex = 6
        # self.urGripperIndex = 7
        # #lower limits for null space
        # self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # #upper limits for null space
        # self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # #joint ranges for null space
        # self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # #restposes for null space
        # self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # #joint damping coefficents
        self.jd = [
            0.000001, 0.1, 10, 10, 0.1, 0.00001
        ]
        self.reset()

    def getGripperJoints(self):
        maxLimit = 0.8
        value = self.degreeOfClosing * maxLimit
        return [value, 0., value, -value, -value, value, -value, 0.]

    def reset(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        # tableID = p.loadURDF("models/table_custom/table.urdf", [0., 0., 0.])
        # p.changeVisualShape(tableID, 1, rgbaColor=[0.466, 0.341, 0.172, 1.0])
        # objects = p.loadSDF(os.path.join(self.urdfRootPath, "kuka_iiwa/kuka_with_gripper2.sdf"))
        objects = [p.loadURDF(self.urdfRootPath + 'ur10/ur10.urdf', [0., 0., 0.58], useFixedBase=1),
                   p.loadURDF(self.urdfRootPath + 'gripper/robotiq_2F85.urdf')]
        self.urUid = objects[0]
        self.gripperUid = objects[1]
        # for i in range (p.getNumJoints(self.kukaUid)):
        #  print(p.getJointInfo(self.kukaUid,i))
        for link_robot in range(p.getNumJoints(self.urUid)):
            p.changeVisualShape(self.urUid, link_robot, rgbaColor=[0.721, 0.831, 0.878, 1.0])
        for link_gripper in range(p.getNumJoints(self.gripperUid)):
            p.changeVisualShape(self.gripperUid, link_gripper, rgbaColor=[0.239, 0.266, 0.282, 1.])

        # p.resetBasePositionAndOrientation(self.urUid, [-0.100000, 0.000000, 0.070000],
        #                                   [0.000000, 0.000000, 0.000000, 1.000000])
        # self.jointPositions = [
        #     0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
        #     -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        # ]

        self.jointPositions = [3.14, -1.57, -1.57, 1.570793, 1.570793, 1.570793, 0.0, 0.]
        self.urNumJoints = p.getNumJoints(self.urUid)
        for jointIndex in range(self.urNumJoints):
            p.resetJointState(self.urUid, jointIndex, self.jointPositions[jointIndex])
        #     p.setJointMotorControl2(self.urUid, jointIndex, p.POSITION_CONTROL, self.jointPositions[jointIndex], 0)
        # p.setJointMotorControlArray(self.urUid, range(8), p.POSITION_CONTROL,
        #                             self.jointPositions)

        ee_state = p.getLinkState(self.urUid, 6)
        print(f'ee pos: {ee_state[0]}\nee orn: {p.getEulerFromQuaternion(ee_state[1])}')
        jointRobotInitial = p.getJointStates(self.urUid, range(8))
        print(f'joints before kinematics: ({jointRobotInitial[0][0]},'
              f'{jointRobotInitial[1][0]}, {jointRobotInitial[2][0]},'
              f'{jointRobotInitial[3][0]}, {jointRobotInitial[4][0]},'
              f'{jointRobotInitial[5][0]}, {jointRobotInitial[6][0]},'
              f'{jointRobotInitial[7][0]})')


        # initial_position_gripper = [0.923103, -0.200000, 1.250036]
        # initial_orientation_gripper = p.getQuaternionFromEuler([0., 0., 0.])
        initial_position_gripper = [ee_state[0][0] + 0.12, ee_state[0][1], ee_state[0][2]]
        initial_orientation_gripper = p.getQuaternionFromEuler([1.57, 0, 1.57])

        p.resetBasePositionAndOrientation(self.gripperUid, initial_position_gripper, initial_orientation_gripper)



        # jointInitialPositionsGripper = [
        #     0.000000, -0.011130, -0.206421, 0.205143, -0.009999, 0.000000, -0.010055, 0.000000
        # ]
        joints = [0, 0, 0, 0, 0, 0, 0, 0]
        for jointIndex in range(p.getNumJoints(self.gripperUid)):
            p.resetJointState(self.gripperUid, jointIndex, joints[jointIndex])
        p.setJointMotorControlArray(self.gripperUid, range(8), p.POSITION_CONTROL, joints)

        parentFrameOrientation = p.getQuaternionFromEuler([0., 0., 0])
        childFrameOrientation = p.getQuaternionFromEuler([1.57, 0, 0])
        self.constraintGripper = p.createConstraint(self.urUid, 6,
                                                    self.gripperUid, -1,
                                                    p.JOINT_FIXED,
                                                    jointAxis=[0, 0, 1],
                                                    parentFramePosition=[0.0, 0.12, 0.0],
                                                    childFramePosition=[0, 0, 0],
                                                    parentFrameOrientation=parentFrameOrientation,
                                                    childFrameOrientation=childFrameOrientation)
        info_constrain = p.getConstraintInfo(self.constraintGripper)
        # print(f'info constraint: force {info_constrain[10]}\njointAxis: {info_constrain[5]}')

        # newOrien = [ee_state[1][0], ee_state[1][1], ee_state[1][2]]
        # jointRobot = p.calculateInverseKinematics(self.urUid, 6,
        #                                           ee_state[0], newOrien)
        # jointRobot = list(jointRobot)
        # jointRobot.insert(0, 0.0)
        # jointRobot.append(0.0)
        # jointRobot[6] += 1.57
        # print(f'joints after kinematics: {jointRobot}')

        # p.setJointMotorControlArray(self.urUid, range(8), p.POSITION_CONTROL, jointRobot)


        self.endEffectorPos = list(ee_state[0])
        robotEndEffectorState = p.getJointState(self.urUid, self.urEndEffectorIndex)
        self.endEffectorAngle = robotEndEffectorState[0]
        self.degreeOfClosing = 0.

        self.motorNames = []
        self.motorIndices = []

        # ee_state = p.getLinkState(self.urUid, 6)
        # print(f'ee pos: {ee_state[0]}\nee orn: {p.getEulerFromQuaternion(ee_state[1])}')

        for i in range(self.urNumJoints):
            jointInfo = p.getJointInfo(self.urUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                # print("motorname")
                # print(jointInfo[1])
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

    def getActionDimension(self):
        if (self.useInverseKinematics):
            return len(self.motorIndices)
        return 6  # position x,y,z and roll/pitch/yaw euler angles of end effector

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        state = p.getLinkState(self.urUid, self.urEndEffectorIndex)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation

    def applyAction(self, motorCommands):
        # motor commands:
        # graspAction = [0, 0, 0.0001, 0, 100]
        jointPoses = []

        # print ("self.numJoints")
        # print (self.numJoints)
        if (self.useInverseKinematics):

            dx = motorCommands[0]
            dy = motorCommands[1]
            dz = motorCommands[2]
            da = motorCommands[3]
            self.degreeOfClosing = motorCommands[4]

            state = p.getLinkState(self.urUid, self.urEndEffectorIndex)
            actualEndEffectorPos = state[0]
            # print("pos[2] (getLinkState(kukaEndEffectorIndex)")
            # print(actualEndEffectorPos[2])

            self.endEffectorPos[0] = self.endEffectorPos[0] + dx
            if (self.endEffectorPos[0] > 1.2):
                self.endEffectorPos[0] = 1.2
            if (self.endEffectorPos[0] < -1.2):
                self.endEffectorPos[0] = -1.2

            self.endEffectorPos[1] = self.endEffectorPos[1] + dy
            if (self.endEffectorPos[1] < -1.2):
                self.endEffectorPos[1] = -1.2
            if (self.endEffectorPos[1] > 1.2):
                self.endEffectorPos[1] = 1.2

            # print ("self.endEffectorPos[2]")
            # print (self.endEffectorPos[2])
            # print("actualEndEffectorPos[2]")
            # print(actualEndEffectorPos[2])
            # if (dz<0 or actualEndEffectorPos[2]<0.5):
            self.endEffectorPos[2] = self.endEffectorPos[2] + dz

            if (self.endEffectorPos[2] < 0.60):
                self.endEffectorPos[2] = 0.60
            if (self.endEffectorPos[2] > 1.2):
                self.endEffectorPos[2] = 1.2

            # self.degreeOfClosing = self.degreeOfClosing + da
            self.endEffectorAngle = self.endEffectorAngle + da
            pos = self.endEffectorPos
            orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
            if (self.useNullSpace == 1):
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.urUid, self.urEndEffectorIndex, pos, orn)
                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)
                    jointPoses[self.urEndEffectorIndex] = self.endEffectorAngle
                else:
                    jointPoses = p.calculateInverseKinematics(self.urUid,
                                                              self.urEndEffectorIndex,
                                                              pos)

                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)
                    jointPoses[self.urEndEffectorIndex] = self.endEffectorAngle
            else:
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.urUid,
                                                              self.urEndEffectorIndex,
                                                              pos, orn,
                                                              jointDamping=self.jd
                                                              )
                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)
                    jointPoses[self.urEndEffectorIndex] = self.endEffectorAngle
                else:
                    jointPoses = p.calculateInverseKinematics(self.urUid, self.urEndEffectorIndex, pos)
                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)
                    jointPoses[self.urEndEffectorIndex] = self.endEffectorAngle

            # print("jointPoses")
            # print(jointPoses)
            # print("self.kukaEndEffectorIndex")
            # print(self.kukaEndEffectorIndex)
            if (self.useSimulation):
                # for i in range(self.urEndEffectorIndex + 1):
                #     # print(i)
                #     p.setJointMotorControl2(bodyUniqueId=self.urUid,
                #                             jointIndex=i,
                #                             controlMode=p.POSITION_CONTROL,
                #                             targetPosition=jointPoses[i],
                #                             targetVelocity=0,
                #                             force=self.maxForce,
                #                             maxVelocity=self.maxVelocity,
                #                             positionGain=0.3,
                #                             velocityGain=1)
                p.setJointMotorControlArray(self.urUid, range(8),
                                            p.POSITION_CONTROL,
                                            jointPoses)
            else:
                # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.urNumJoints):
                    p.resetJointState(self.urUid, i, jointPoses[i])
            # fingers
            jointPosesGripper = self.getGripperJoints()
            p.setJointMotorControlArray(self.gripperUid, range(8), p.POSITION_CONTROL,
                                        jointPosesGripper)

        else:
            print('stange action, do not get')
            for action in range(len(motorCommands)):
                motor = self.motorIndices[action]
                p.setJointMotorControl2(self.urUid,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=motorCommands[action],
                                        force=self.maxForce)
