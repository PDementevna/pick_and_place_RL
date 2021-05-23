# Deep Reinforcement Learning on Robotics Grasping

The task is to get a model for a successful pick and place the cube by the robot manipulator. 
The system consists of the UR10 robot with an appropriate gripper (ROBOTIQ 2f-85) and a randomly 
appearing cube on the table. The framework for simulation the robot behavior is PyBullet. The 
algorithm for RL training is PPO2 from stable baselines.


### Prerequisites

Install anaconda. Start a clean conda environment.

`conda create -n bullet python=3.7`

`conda activate bullet`

## Installation

Choose the location for the project and move there. 
Then clone the project from GitHub

`git clone https://github.com/PDementevna/pick_and_place_RL.git`

Move to that folder:

`cd pick_and_place_RL`

Install necessary packages for python:

`pip install -r requirements.txt`

Install the custom environment:

`pip install -e ur-env`

## Train

For training the model, it is possible to  launch a training script
with adjusted parameters:

`python train_ur_grasping.py`

## Test

For testing the model, launch the python file:

`python enjoy_ur_grasping.py`

It will use weights from the best model. To switch between model, use any file with weights for 
testing by specifying the name of the file (all the trained model listed in **./trained_models** folder).
Example of using file for loading the model with name _ppo2_trial1_10e7.zip_:

`model = PPO2.load("trained_models/ppo2_trial1_10e7")`

## Test the environment

For familiarization with the environment, launch the python file:

`python URGymEnvTest.py`

## Discussion of results

The best performance was obtained for model `ppo2_trial16_10e7`. 
The model is trained for the environment where the cube appears in a small area (0.1 x 0.1) without any
rotation around the z-axis. On each step, the model predicts action with dimension 2 for x coordinate
and y coordinate of robot's end-effector while z-axis is slowly decreasing on -0.005 on each step.
The reward is increasing by gripper approaching the cube center position from the point between the
gripper fingers, and the additional reward in 10k is adding when the z coordinate of the cube is
higher than 0.1 from the table (related to the initial position). The process's termination starts
when the distance between the cube center and the point between the gripper finger is less than 0.01005.
The termination process includes 4 stages: gripper closing (to take the cube), gripper lifting (to lift
the cube), gripper moving (to reach the position for placing the cube: the tray center), gripper opening
(to place the cube). The graph of episode rewards obtained over time steps is presented below.


<img src="https://github.com/PDementevna/pick_and_place_RL/raw/env_with_orn/episode_reward.svg" height="300">

The test process was made on a greater area (0.3 x 0.5) and for 2 configurations of the environment:
1) the cube remains with 0 angle rotation around the z-axis, 2) the cube obtained the random angle around
   the z-axis (from 0 to pi/2). The results of model prediction performance for these 2 configurations
   are presented below:


### The env configuration without cube rotation

<img src="https://github.com/PDementevna/pick_and_place_RL/raw/env_with_orn/video/withoutOrn.gif" height="400">

### The env configuration with cube rotation

<img src="https://github.com/PDementevna/pick_and_place_RL/raw/env_with_orn/video/withOrn.gif" height="400">

### Evaluation of the model performance

1000 iterations of launching each of the environment configuration were used to evaluate the model
performance. For the evaluation, 3 states of the environment (during the termination process) were
used: the gripper fingers touched the cube (**isCaught**), the gripper lifted the cube on the required
z position (**isLifted**), and the gripper placed the cube into the tray (**isMoved**). It means that
the isCaught metric is based on 1000 possible successful efforts, the isLifted based on the number of
isCaught successful attempts, and the isMoved is based on the number of isLifted successful attempts.
The evaluation for the whole process of "pick and place" task is also provided. 


#### The env configuration without cube rotation
Stage | Number of attempts | Successful attempts | % (success)
---| --- | --- | ---
isCaught | 1000 | 998 | 99.8
isLifted | 998 | 796 | 79.8
isMoved  | 796 | 608 | 76.4

The pick and place task performance for env configuration without cube rotation 
is 608 out of 1000: **60.8**%

#### The env configuration with cube rotation
Stage | Number of attempts | Successful attempts | % (success)
---| --- | --- | ---
isCaught | 1000 | 996 | 96.6
isLifted | 996 | 741 | 74.4
isMoved  | 741 | 539 | 72.7

The pick and place task performance for env configuration with cube rotation 
is 539 out of 1000 : **53.9**%

#### Example of unsuccessful attempt to pick and place the cube

<img src="https://github.com/PDementevna/pick_and_place_RL/raw/env_with_orn/video/unsuccessful.gif" height="400">
