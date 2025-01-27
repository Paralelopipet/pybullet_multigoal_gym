import matplotlib.pyplot as plt

import pybullet_multigoal_gym as pmg
from pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs import \
    KukaTipOverEnv

f, axarr = plt.subplots(1, 2)

env: KukaTipOverEnv = pmg.make_env(task='tip_over',
                   gripper='parallel_jaw_cube',
                   render=True,
                   binary_reward=True,
                   joint_control=True,
                   max_episode_steps=50,
                   image_observation=False,
                   depth_image=False,
                   goal_image=False,
                   visualize_target=True,
                   camera_setup=None,
                   observation_cam_id=[0],
                   goal_cam_id=0,
                   target_range=0.3,
                   plane_position = [0.,0.,-0.58],
                   )
obs = env.reset()
t = 0
done = False
while True:
    t += 1
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # axarr[0].imshow(obs['desired_goal_img'])
    # axarr[1].imshow(obs['achieved_goal_img'])
    # plt.pause(0.0001)
    if done:
        env.reset()
