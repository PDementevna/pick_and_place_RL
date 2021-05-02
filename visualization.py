import pybullet as p
import time
import pybullet_data
import numpy as np
from ur_env.envs import ur


def get_random_position(outerLimitation, innerLimitation, z_coord=0.58):
    x_pos = np.random.rand() * outerLimitation[1] + outerLimitation[0]
    y_pos = np.random.rand() * outerLimitation[1] + outerLimitation[0]

    if internalLim[1] > x_pos > innerLimitation[0]:
        get_random_position(outerLimitation, innerLimitation)
    if internalLim[1] > y_pos > innerLimitation[0]:
        get_random_position(outerLimitation, innerLimitation)
    return [x_pos, y_pos, z_coord]


physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

# p.setGravity(0, 0, -9.81)

planeId = p.loadSDF('stadium.sdf')
# cubeStartPos = [0,0,1]
# cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
tableID = p.loadURDF("models/table_custom/table.urdf", [0., 0., 0.])
p.changeVisualShape(tableID, 1, rgbaColor=[0.466, 0.341, 0.172, 1.0])

_ur = ur.UR()

# robotID = p.loadURDF("models/ur10_gripper/ur10_gripper.urdf",
#                      [0., 0., 0.58], useFixedBase=1)

# UNCOMMENT BELOW

# robotID = p.loadURDF('models/ur10/ur10.urdf', [0., 0., 0.58], useFixedBase=1)
# gripperID = p.loadURDF('models/gripper/robotiq_2F85.urdf')
#
# for link_robot in range(p.getNumJoints(robotID)):
#     p.changeVisualShape(robotID, link_robot, rgbaColor=[0.721, 0.831, 0.878, 1.0])
# for link_gripper in range(p.getNumJoints(gripperID)):
#     p.changeVisualShape(gripperID, link_gripper, rgbaColor=[0.239, 0.266, 0.282, 1.])
#
# jointPositions = [0.000000, -1.57, -1.57, 1.570793, 0.000000, 0.0, 0.0, 0.]
# for jointIndex in range(p.getNumJoints(robotID)):
#   p.resetJointState(robotID, jointIndex, jointPositions[jointIndex])
#   p.setJointMotorControl2(robotID, jointIndex, p.POSITION_CONTROL, jointPositions[jointIndex], 0)
#
# initial_position_gripper = [0.923103, -0.200000, 1.250036]
# initial_orientation_gripper = p.getQuaternionFromEuler([0., 0., 0.])
#
# p.resetBasePositionAndOrientation(gripperID, initial_position_gripper, initial_orientation_gripper)
# jointPositions = [
#     0.000000, -0.011130, -0.206421, 0.205143, -0.009999, 0.000000, -0.010055, 0.000000
# ]
# for jointIndex in range(p.getNumJoints(gripperID)):
#   p.resetJointState(gripperID, jointIndex, jointPositions[jointIndex])
#   p.setJointMotorControl2(gripperID, jointIndex, p.POSITION_CONTROL, jointPositions[jointIndex],
#                           0)
#
outerLim = [-1.0, 2.0]
internalLim = [-0.15, 0.30]
# z_lim = 0.70
# count = 0
#
# parentFrameOrientation = p.getQuaternionFromEuler([0., 0., 0])
# childFrameOrientation = p.getQuaternionFromEuler([1.57, 0, 1.57])
# kuka_cid = p.createConstraint(robotID, 6, gripperID, 0, p.JOINT_FIXED, [0, 0, 0], [0.0, 0.16, 0.05],
#                               [0, 0, 0], parentFrameOrientation, childFrameOrientation)


# UNCOMMENT ABOVE


# nJointsRobot = p.getNumJoints(robotID)
#
# jointNameToIdRobot = {}
# for i in range(nJointsRobot):
#     # print(f'i: {i}')
#     jointInfoRobot = p.getJointInfo(robotID, i)
#     jointNameToIdRobot[jointInfoRobot[1].decode('UTF-8')] = jointInfoRobot[0]
#
# nJointsGripper = p.getNumJoints(gripperID)
#
# jointNameToIdGripper = {}
# for i in range(nJointsGripper):
#     jointInfoGripper = p.getJointInfo(gripperID, i)
#     jointNameToIdGripper[jointInfoGripper[1].decode('UTF-8')] = jointInfoGripper[0]

# print(*jointNameToIdGripper)
# link_index_robot = jointNameToIdRobot['ee_fixed_joint']
# link_index_gripper = jointNameToIdGripper['finger_joint']
#
#
# position, orientation = p.getBasePositionAndOrientation(robotID)  # orientation is in quanterion
# cid = p.createConstraint(robotID, link_index_robot, gripperID, link_index_gripper, p.JOINT_POINT2POINT, [0, 0, 0],
#                          [0, 0.005, 0.1], [0, 0.01, 0.1], orientation, orientation)


# print(p.getNumJoints(robotID))
# print(position)

# robotID = p.loadURDF("ur5/ur5.urdf")



# UNCOMMENT BELOW
# orient = p.getQuaternionFromEuler([0., 0., 0.])
# UNCOMMENT ABOVE




# targetPositionsJoints = p.calculateInverseKinematics(robotID, 6, [0.5, 0.5, -1.58], targetOrientation=orient)
# p.setJointMotorControlArray(robotID, range(6), p.POSITION_CONTROL, targetPositions=targetPositionsJoints)

# targetPositionsJoints = p.calculateInverseKinematics(robotID, 16, [0.5, 0.5, 0.5])
# p.setJointMotorControlArray(robotID, range(16), p.POSITION_CONTROL, targetPositions=targetPositionsJoints)
viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[0, 0, 3],
    cameraTargetPosition=[0, 0, 1],
    cameraUpVector=[0, 1, 0])

projectionMatrix = p.computeProjectionMatrixFOV(
    fov=60.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=3.1)

# cubeStartPos = get_random_position(outerLim, internalLim)
# cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# cubeID = p.loadURDF("cube_small.urdf", cubeStartPos, cubeStartOrientation)

# Camera paramers to be able to yaw pitch and zoom the camera (Focus remains on the robot)
cyaw = 30
cpitch = -50
cdist = 1.3

# cyaw = 130
# cpitch = -30
# cdist = 3


# UNCOMMENT BELOW
# cubeStartPos = get_random_position(outerLim, internalLim, 0.65)
# cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# # cubeID = p.loadURDF("cube_small.urdf", cubeStartPos, cubeStartOrientation)
# cubeID = p.loadURDF("cube_small.urdf", [0.7, 0.7, 0.60])
# UNCOMMENT ABOVE

# closed_gripper_joints = [0.725, ]

# targetPositionsJoints = p.calculateInverseKinematics(robotID, 6, [0.7, 0.7, -0.60])
# p.setJointMotorControlArray(robotID, range(6), p.POSITION_CONTROL, targetPositions=targetPositionsJoints)

while True:
    # if count % 5 == 0:
    #     cubeStartPos = get_random_position(outerLim, internalLim, 0.65)
    #     cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    #     cubeID = p.loadURDF("cube_small.urdf", cubeStartPos, cubeStartOrientation)
    #
    #     print(f'position cube: {cubeStartPos[0]}, {cubeStartPos[1]}')
    #     count = 0

    # state = p.getJointState(robotID, 6)
    # print(f'pos joint: {state[0]}')

    pos, orn = p.getBasePositionAndOrientation(_ur.urUid)
    p.resetDebugVisualizerCamera(cameraDistance=cdist, cameraYaw=cyaw, cameraPitch=cpitch, cameraTargetPosition=pos)

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=224,
        height=224,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)

    keys = p.getKeyboardEvents()
    # Keys to change camera
    if keys.get(100):  # D
        cyaw += 5
    if keys.get(97):  # A
        cyaw -= 5
    if keys.get(99):  # C
        cpitch += 5
    if keys.get(102):  # F
        cpitch -= 5
    if keys.get(122):  # Z
        cdist += .1
    if keys.get(120):  # X
        cdist -= .1

    p.stepSimulation()
    # count += 1
    time.sleep(0.001)
    jointPositionsGripper = [
        0.8757, -0.8757, 0.8757, -0.8757, -0.8757, 0.8757, -0.8757, 0.8757
    ]
    # p.setJointMotorControlArray(_ur.gripperUid, range(8), p.POSITION_CONTROL, jointPositionsGripper)

# for _ in range(300):
#     p.stepSimulation()
#     width, height, rgbImg, depthImg, segImg = p.getCameraImage(
#         width=224,
#         height=224,
#         viewMatrix=viewMatrix,
#         projectionMatrix=projectionMatrix)
#     time.sleep(1. / 10.)

p.disconnect()
