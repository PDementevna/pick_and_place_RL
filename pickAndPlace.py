import numpy as np


def dist2points(point1, point2):
	return np.linalg.norm(point1 - point2)


def waitForSuccess(condition, threshold=0.001):
	if (dist2points(condition[0], condition[1]) > threshold):
		return True
	return False


def moveToCubeXY(cubePos, p):
	eePos = cubePos
	eePos[2] = 1.0
	# currEEPos = p.

def moveToCubeZ():
	pass

def graspCube():
	pass

def liftCube():
	pass

def placeCube():
	pass

def putCube():
	pass

