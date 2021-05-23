# Deep Reinforcement Learning on Robotics Grasping

The training of the UR10 robot with appropriate gripper for 
picking a randomly appeared cube on the table and placing it 
in a specific place. The framework for simulation the robot behaviour 
is PyBullet. The algorithm for RL training is PPO2 from stable baselines.
The video of the performance is here: https://drive.google.com/file/d/1MEtzKqKblDXoIMLvLFJbaJatPhawon0w/view?usp=sharing

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

It will use weights from the latest model. If you would like to use 
another weights for testing, specify a name of the file (all of the
trained model listed in **./trained_models** folder).
Example of using file for loading the model with name _ppo2_ur5000.zip_:
`model = PPO2.load("trained_models/ppo2_ur5000")`

## Discussion of results

So, the trained models work insufficiently. I had several approaches
for training the model on a custom environment based on KukaEnv from
examples. 
First of all, I used the UR10 robot with Robotiq 2F-85 gripper. 
I tried to combine two urdf files to work with one object, but 
further, I got that it's better to unite them with createConstraint
function. As for the first models, I used the whole table area 
to generate cubes randomly, but the models didn't seem to work well.
Therefore, I reduced the sizes of the working area where the cubes 
will appear to get a result there. I tried a different number of 
actions sent to the system of each step and made the penalties for 
some of the actions to reduce reward. Also added the additional 
reward scheme about picking the object or being around it. But all 
that I have done by now haven't shown precise performance in grasping,
unfortunately.
The actions which I haven't tried yet:
1) Train the only gripper to move and grasp the object, and just on
   the test scene, add the robot to perform the entire action of 
   picking and placing.
2) Try to train the model on different policies to compare results.

As it all new for me, I did my best to achieve the results in grasping 
and it's challenging.
Whatever you decide about me, it was nice to try all those things,
thank you. 


![Alt Text](https://github.com/PDementevna/pick_and_place_RL/blob/env_with_orn/video/withoutOrn.gif | width=100)
