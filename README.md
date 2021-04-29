**Conda**
`conda create -n bullet python=3.7`
`conda activate bullet`

Then we need to decide the location of our project, move to the chosen folder and then clone the project:
`git clone https://github.com/PDementevna/pick_and_place_RL.git`

Install necessary packages for python:
`pip install tensorflow-gpu==1.15 pybullet stable-baselines`


To register new custom environment for using RL, you need to use following command:

`pip install -e ur-env`
