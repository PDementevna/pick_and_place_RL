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
        maxLimit = 0.8
        if (close):
            value = self.degreeOfClosing * maxLimit
        else:
            value = 0.
        return [value, 0., value, -value, -value, value, -value, 0.]

    def _getGripperPosLink(self, num_link):
        link_state = p.getLinkState(self.gripperUid, num_link)
        return link_state[0]

    def getEEGripperPos(self):
        finger1Pos = self._getGripperPosLink()

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


        # self.jointPositions = [3.14, -1.57, -1.57, 1.570793, -1.57, -1.57, 0.0, 0.]
        ee_position = [0.45, 0.1, 1.00]

        ee_orin = [math.pi / 2.0, math.pi, -math.pi / 2.]
        # print(f'pi/2 = {math.pi / 2.}')
        # ee_orin = [math.pi / 2., math.pi , 0]
        ee_angle = p.getQuaternionFromEuler(ee_orin)
        jointPositions = p.calculateInverseKinematics(self.urUid,
                                                  self.urEndEffectorIndex,
                                                  ee_position,
                                                  ee_angle,
                                                  # jointDamping=self.jd
                                                  )



        jointPositions = list(jointPositions)
        jointPositions.insert(0, 0.0)
        jointPositions.append(0.0)

        # self.jointPositions = jointPositions
        self.jointPositions = jointPositions
        # self.jointPositions = [0.0, 0.5537298288314505, -1.1481937571631586, 1.5561487412923882, 1.093472799428354, 0., 0.0, 0.0]
        #                       [0.0, 0.5537298288314505, -1.1481937571631586, 1.5561487412923882, 1.093472799428354, 0.0, 0.0, 0.0]
        #                       [0.0, 0.5537298288314505, -1.1481937571631586, 1.5561487412923882, 1.093472799428354, 0.0, 0.0, 0.0]
        # final:                [0.0, 0.48002757997666934, -1.4129673165748495, 1.9689138336025802, 4.2199519480313885, -1.5707963267948966, 0.0, 0.0]
        # for 0.45, 0.1, 1.0   mipu9- [0.0, -0.055012701204166244, -1.3261254429073293, 1.8364445377448748, 4.2377985733831105, -1.5707963267948966, 0.0, 0.0]



        self.urNumJoints = p.getNumJoints(self.urUid)
        for jointIndex in range(self.urNumJoints):
            p.resetJointState(self.urUid, jointIndex, self.jointPositions[jointIndex])


        ee_state = p.getLinkState(self.urUid, 6)
        # print(f'reset pos mass: {ee_state[0]}')
        # print(f'reset pos world: {ee_state[4]}')
        # print(f'reset local: {ee_state[2]}')

        jointRobotInitial = p.getJointStates(self.urUid, range(8))

        # initial_position_gripper = [ee_state[0][0], ee_state[0][1], ee_state[0][2]]
        # initial_position_gripper = [ee_state[0][0], ee_state[0][1], ee_state[0][2]]
        initial_position_gripper = [ee_state[0][0], ee_state[0][1], ee_state[0][2] - 0.12]
        initial_orientation_gripper = p.getQuaternionFromEuler([math.pi, 0., math.pi / 2.0])

        p.resetBasePositionAndOrientation(self.gripperUid, initial_position_gripper, initial_orientation_gripper)
        # print(f'gripper base pos: {p.getLinkState(self.gripperUid, 0)[0]}')

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


        info_constrain = p.getConstraintInfo(self.constraintGripper)

        # self.endEffectorPos = list(ee_state[0])
        self.endEffectorPos = ee_position

        # print(f'ee pos: {self.endEffectorPos}')
        robotEndEffectorState = p.getLinkState(self.urUid, self.urEndEffectorIndex)

        # self.endEffectorAngle = robotEndEffectorState[1][2]
        self.endEffectorAngle = math.pi / 2.0
        # self.endEffectorAngle = p.getJointState(self.urUid, 6)[0]
        # print(f'angle of ee {self.endEffectorAngle}')
        self.degreeOfClosing = 0.

        self.motorNames = []
        self.motorIndices = []


        for i in range(self.urNumJoints):
            jointInfo = p.getJointInfo(self.urUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                # print("motorname")
                # print(jointInfo[1])
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)


        # print('finished')
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

    def closeGripper(self):
        jointPosesGripper = self.getGripperJoints()
        p.setJointMotorControlArray(self.gripperUid, range(8), p.POSITION_CONTROL,
                                    jointPosesGripper)

    def openGripper(self):
        jointPosesGripper = self.getGripperJoints(close = False)
        p.setJointMotorControlArray(self.gripperUid, range(8), p.POSITION_CONTROL,
                                    jointPosesGripper)

    def printPosLinks(self):
        links_state = p.getLinkStates(self.urUid, range(8))
        for i, link in enumerate(links_state):
            print(f'link {i}, pos: {link[0]}')

    def applyAction(self, motorCommands):
        # print(f'motor commands: {motorCommands}')
        if (self.useInverseKinematics):

            dx = motorCommands[0]
            dy = motorCommands[1]
            dz = motorCommands[2]
            da = motorCommands[3]
            # print(f'action[3]: {motorCommands[3]}')
            self.degreeOfClosing = motorCommands[4]

            state = p.getLinkState(self.urUid, self.urEndEffectorIndex, computeForwardKinematics=True)
            gripperPose = p.getLinkState(self.gripperUid, 0, computeForwardKinematics=True)
            actualEndEffectorPos = state[0]
            # print()
            # print(f'actual ee pos: {actualEndEffectorPos}')
            # print(f'gripper pos: {gripperPose}')
            # self.printPosLinks()
            # print(f'set ee pos: {self.endEffectorPos}')

            self.endEffectorPos[0] = self.endEffectorPos[0] + dx


            if (self.endEffectorPos[0] > 0.52):
                # print('boundary x+')
                self.endEffectorPos[0] = 0.52
            if (self.endEffectorPos[0] < 0.38):
                # print('boundary x-')
                self.endEffectorPos[0] = 0.38

            self.endEffectorPos[1] = self.endEffectorPos[1] + dy
            if (self.endEffectorPos[1] < -0.02):
                # print('boundary y-')
                self.endEffectorPos[1] = -0.02
            if (self.endEffectorPos[1] > 0.13):
                # print('boundary y+')
                self.endEffectorPos[1] = 0.13

            self.endEffectorPos[2] = self.endEffectorPos[2] + dz
            if (self.endEffectorPos[2] < 0.88):
                # print('boundary z-')
                self.endEffectorPos[2] = 0.88
            if (self.endEffectorPos[2] > 1.1):
                # print('boundary z+')
                self.endEffectorPos[2] = 1.1



            self.endEffectorAngle += da
            pos = self.endEffectorPos
            # orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
            # orn = [-math.pi / 2., math.pi / 2., math.pi]
            # orn[1] += self.endEffectorAngle
            # orn = p.getQuaternionFromEuler(orn)
            ee_orin = [math.pi / 2., math.pi, self.endEffectorAngle]
            # print(f'ee orientation: {ee_orin}')
            # print(f'pi/2 = {math.pi / 2.}')
            # ee_orin = [math.pi / 2., math.pi , 0]
            ee_angle = p.getQuaternionFromEuler(ee_orin)
            if (self.useNullSpace == 1):
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.urUid, self.urEndEffectorIndex, pos, ee_angle)
                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)

                    # jointPoses[5] -= math.pi / 2.0
                    # jointPoses[4] += math.pi

                    # jointPoses[6] = self.endEffectorAngle

                else:
                    jointPoses = p.calculateInverseKinematics(self.urUid,
                                                              self.urEndEffectorIndex,
                                                              pos)
                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)

                    # jointPoses[5] -= math.pi / 2.0
                    # jointPoses[4] += math.pi

            else:
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.urUid,
                                                              self.urEndEffectorIndex,
                                                              pos,
                                                              ee_angle
                                                              # jointDamping=self.jd
                                                              )
                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)

                    # jointPoses[5] -= math.pi / 2.0
                    # jointPoses[4] += math.pi
                    #
                    # jointPoses[6] = self.endEffectorAngle
                    # print(f'jointPoses: {jointPoses}')

                else:
                    jointPoses = p.calculateInverseKinematics(self.urUid, self.urEndEffectorIndex, pos)
                    jointPoses = list(jointPoses)
                    jointPoses.insert(0, 0.0)
                    jointPoses.append(0.0)

                    # jointPoses[5] -= math.pi / 2.0
                    # jointPoses[4] += math.pi

            if (self.useSimulation):
                p.setJointMotorControlArray(self.urUid, range(8),
                                            p.POSITION_CONTROL,
                                            jointPoses)
            else:
                # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.urNumJoints):
                    p.resetJointState(self.urUid, i, jointPoses[i])
            # fingers
            jointPosesGripper = self.getGripperJoints()
            # print(f'len of joints gripper: {jointPosesGripper}')
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
