import pybullet as p
import math
import pybullet_data


class UR:

    def __init__(self, urdfRootPath='models/', timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.fingerTipForce = 2
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 21
        self.useOrientation = 1
        self.urEndEffectorIndex = 6
        self.jd = [
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        ]
        self.reset()

    def getGripperJoints(self, close=True):
        maxLimit = 1.2
        if (close):
            value = self.degreeOfClosing * maxLimit
        else:
            value = 0.
        return [value - value / 2.,
                0.,
                value - value / 2.,
                0,
                -value + value / 2.,
                0,
                -value + value / 2.,
                0.]

    def reset(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        robot = p.loadURDF(self.urdfRootPath + 'ur10/ur10.urdf', [0., 0., 0.58], useFixedBase=1)

        gripper = p.loadURDF(self.urdfRootPath + 'gripper/robotiq_2F85.urdf')
        self.urUid = robot
        self.gripperUid = gripper
        for link_robot in range(p.getNumJoints(self.urUid)):
            p.changeVisualShape(self.urUid, link_robot, rgbaColor=[0.721, 0.831, 0.878, 1.0])
        for link_gripper in range(p.getNumJoints(self.gripperUid)):
            p.changeVisualShape(self.gripperUid, link_gripper, rgbaColor=[0.239, 0.266, 0.282, 1.])

        ee_position = [0.45, 0.1, 1.00]

        ee_orin = [math.pi / 2.0, math.pi, -math.pi / 2.]

        ee_angle = p.getQuaternionFromEuler(ee_orin)
        jointPositions = p.calculateInverseKinematics(self.urUid,
                                                  self.urEndEffectorIndex,
                                                  ee_position,
                                                  ee_angle,
                                                  )



        jointPositions = list(jointPositions)
        jointPositions.insert(0, 0.0)
        jointPositions.append(0.0)

        self.jointPositions = jointPositions

        self.urNumJoints = p.getNumJoints(self.urUid)
        for jointIndex in range(self.urNumJoints):
            p.resetJointState(self.urUid, jointIndex, self.jointPositions[jointIndex])

        ee_state = p.getLinkState(self.urUid, 6)

        initial_position_gripper = [ee_state[0][0], ee_state[0][1], ee_state[0][2] - 0.12]
        initial_orientation_gripper = p.getQuaternionFromEuler([math.pi, 0., math.pi / 2.0])

        p.resetBasePositionAndOrientation(self.gripperUid, initial_position_gripper, initial_orientation_gripper)

        joints = [0, 0, 0, 0, 0, 0, 0, 0]
        for jointIndex in range(p.getNumJoints(self.gripperUid)):
            p.resetJointState(self.gripperUid, jointIndex, joints[jointIndex])
        p.setJointMotorControlArray(self.gripperUid, range(8), p.POSITION_CONTROL, joints)

        parentFrameOrientation = p.getQuaternionFromEuler([0., 0., 0])
        childFrameOrientation = p.getQuaternionFromEuler([math.pi / 2., 0, 0])


        self.constraintGripper = p.createConstraint(self.urUid, 6,
                                                    self.gripperUid, -1,
                                                    p.JOINT_FIXED,
                                                    jointAxis=[0, 0, 1],
                                                    parentFramePosition=[0.0, 0.12, 0.0],
                                                    childFramePosition=[0, 0, 0],
                                                    parentFrameOrientation=parentFrameOrientation,
                                                    childFrameOrientation=childFrameOrientation)

        self.endEffectorPos = ee_position
        self.endEffectorAngle = math.pi / 2.0
        self.degreeOfClosing = 0.
        self.motorNames = []
        self.motorIndices = []


        for i in range(self.urNumJoints):
            jointInfo = p.getJointInfo(self.urUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
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
        if (self.useInverseKinematics):

            dx = motorCommands[0]
            dy = motorCommands[1]
            dz = motorCommands[2]
            da = motorCommands[3]
            self.degreeOfClosing = motorCommands[4]
            self.endEffectorPos[0] = self.endEffectorPos[0] + dx


            # if (self.endEffectorPos[0] > 0.62):
            #     # print('boundary x+')
            #     self.endEffectorPos[0] = 0.62
            # if (self.endEffectorPos[0] < 0.38):
            #     # print('boundary x-')
            #     self.endEffectorPos[0] = 0.38

            self.endEffectorPos[1] = self.endEffectorPos[1] + dy
            # if (self.endEffectorPos[1] < -0.30):
            #     # print('boundary y-')
            #     self.endEffectorPos[1] = -0.30
            # if (self.endEffectorPos[1] > 0.3):
            #     # print('boundary y+')
            #     self.endEffectorPos[1] = 0.3

            self.endEffectorPos[2] = self.endEffectorPos[2] + dz
            if (self.endEffectorPos[2] < 0.90):
                # print('boundary z-')
                self.endEffectorPos[2] = 0.90
            if (self.endEffectorPos[2] > 1.3):
                # print('boundary z+')
                self.endEffectorPos[2] = 1.3

            self.endEffectorAngle += da
            pos = self.endEffectorPos
            ee_orin = [math.pi / 2., math.pi, self.endEffectorAngle]
            ee_angle = p.getQuaternionFromEuler(ee_orin)
            if (self.useNullSpace == 1):
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.urUid, self.urEndEffectorIndex, pos, ee_angle)
                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)

                else:
                    jointPoses = p.calculateInverseKinematics(self.urUid,
                                                              self.urEndEffectorIndex,
                                                              pos)
                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)

            else:
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.urUid,
                                                              self.urEndEffectorIndex,
                                                              pos,
                                                              ee_angle
                                                              )
                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)

                else:
                    jointPoses = p.calculateInverseKinematics(self.urUid, self.urEndEffectorIndex, pos)
                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)

            if (self.useSimulation):
                p.setJointMotorControlArray(self.urUid, range(8),
                                            p.POSITION_CONTROL,
                                            jointPoses)
            else:
                for i in range(self.urNumJoints):
                    p.resetJointState(self.urUid, i, jointPoses[i])
            # fingers
            jointPosesGripper = self.getGripperJoints()
            force = 100
            p.setJointMotorControlArray(self.gripperUid, range(8), p.POSITION_CONTROL,
                                        jointPosesGripper, forces=[force, force, force, force, force, force, force, force])

        else:
            for action in range(len(motorCommands)):
                motor = self.motorIndices[action]
                p.setJointMotorControl2(self.urUid,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=motorCommands[action])
