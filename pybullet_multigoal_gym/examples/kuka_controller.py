import matplotlib.pyplot as plt

import pybullet_multigoal_gym as pmg
from pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs import \
    KukaTipOverEnv
import time
from seer.controller import ForceAngleController
from seer.stability_metrics.adapter import StabilityMetricAdapter, RobotConfig, RobotState
from seer.stability_metrics.force_angle import ForceAngleConfigAdapter, ForceAngle
import numpy as np
from seer.environment import GymEnvironment
from seer.trajectory import TestTrajectory, SimpleTrajectory
import time 
f, axarr = plt.subplots(1, 2)

env: KukaTipOverEnv = pmg.make_env(task='tip_over',
                   gripper='parallel_jaw_cube',
                   render=True,
                   binary_reward=True,
                   joint_control=False,
                   max_episode_steps=50,
                   image_observation=False,
                   depth_image=False,
                   goal_image=False,
                   visualize_target=True,
                   camera_setup=None,
                   observation_cam_id=[0],
                   goal_cam_id=0,
                   target_range=0.3,
                   )

obs = env.reset()
t = 0
done = False
controllerEnv =  GymEnvironment(env)
controller = ForceAngleController(controllerEnv)
# testTrajectory = TestTrajectory(None, None)
# testTrajectory = SimpleTrajectory(controller.getEndEffectorWorldPosition(), [0,0,1.5], time.time())
testTrajectory = SimpleTrajectory(controller.getEndEffectorWorldPosition(), controller.getEndEffectorWorldPosition(), time.time())
controller.setTrajectory(testTrajectory)

while True:
    t += 1
    # action = controller.getNextJointPositions()
    action = controller.trajectory.getPosition(time.time())
    action = np.array(action)
    action = np.clip(action, -1, 1)

    print(action)
    obs, reward, done, info = env.step(action)
    # axarr[0].imshow(obs['desired_goal_img'])
    # axarr[1].imshow(obs['achieved_goal_img'])
    # plt.pause(0.0001)
    # if done:
    #     env.reset()