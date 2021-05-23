# Deep Reinforcement Learning on Robotics Grasping

The training of the UR10 robot with appropriate gripper for 
picking a randomly appeared cube on the table and placing it 
in a specific place. The framework for simulation the robot behaviour 
is PyBullet. The algorithm for RL training is PPO2 from stable baselines.

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

## Train

For training the model, you can launch training script
with adjusted parameters:

`python train_ur_grasping.py`

## Test

For testing the model, launch the python file:

`python enjoy_ur_grasping.py`

It will use weights from the best model. If you would like to use 
another weights for testing, specify a name of the file (all the
trained model listed in **./trained_models** folder).
Example of using file for loading the model with name _ppo2_trial1_10e7.zip_:
`model = PPO2.load("trained_models/ppo2_ur5000")`

## Discussion of results

<img src="https://github.com/PDementevna/pick_and_place_RL/raw/env_with_orn/episode_reward.svg" height="400">

<img src="https://github.com/PDementevna/pick_and_place_RL/raw/env_with_orn/video/withoutOrn.gif" height="400">
