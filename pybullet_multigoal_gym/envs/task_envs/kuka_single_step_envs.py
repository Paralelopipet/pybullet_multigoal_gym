from pybullet_multigoal_gym.envs.base_envs.kuka_single_step_base_env import KukaBulletMGEnv
from pybullet_multigoal_gym.envs.base_envs.kuka_single_3_joint_env import KukaBullet3Env


class KukaPickAndPlaceEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, joint_control=False, distance_threshold=0.05,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                 image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                 visualize_target=visualize_target,
                                 camera_setup=camera_setup, observation_cam_id=observation_cam_id,
                                 goal_cam_id=goal_cam_id,
                                 gripper_type=gripper_type, obj_range=0.15, target_range=0.15,
                                 target_in_the_air=True,
                                 grasping=True, joint_control=joint_control, has_obj=True)


class KukaPushEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, joint_control=False, distance_threshold=0.05,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                 image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                 visualize_target=visualize_target,
                                 camera_setup=camera_setup, observation_cam_id=observation_cam_id,
                                 goal_cam_id=goal_cam_id,
                                 gripper_type=gripper_type, obj_range=0.15, target_range=0.15,
                                 target_in_the_air=False, end_effector_start_on_table=True,
                                 grasping=False, joint_control=joint_control, has_obj=True)


class KukaReachEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, joint_control=False, distance_threshold=0.05,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0,
                 gripper_type='parallel_jaw'):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                 image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                 visualize_target=visualize_target,
                                 camera_setup=camera_setup, observation_cam_id=observation_cam_id,
                                 goal_cam_id=goal_cam_id,
                                 gripper_type=gripper_type, obj_range=0.15, target_range=0.15,
                                 target_in_the_air=True,
                                 grasping=False, joint_control=joint_control, has_obj=False)


class KukaTipOverEnv(KukaBullet3Env):
    def __init__(self, render=True, binary_reward=True, gravity_angle=0.0, joint_control=True, distance_threshold=0.05,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0, target_range=0.15, plane_position=[0., 0., -1.], has_spring=False,  joint_force_sensors=False,
                 gripper_type='parallel_jaw', tip_penalty=-10, tipping_threshold=0.5, force_angle_reward_factor=15, noise_stds={}, target_min_distance=0.1, target_min_distance_xy=0.1,checkReachability=True):
        KukaBullet3Env.__init__(self, render=render, binary_reward=binary_reward, gravity_angle=gravity_angle, distance_threshold=distance_threshold,
                                image_observation=image_observation, goal_image=goal_image, depth_image=depth_image,
                                visualize_target=visualize_target,
                                camera_setup=camera_setup, observation_cam_id=observation_cam_id,
                                goal_cam_id=goal_cam_id,
                                gripper_type=gripper_type, obj_range=0.15, target_range=target_range,
                                plane_position=plane_position, has_spring=has_spring, joint_force_sensors=joint_force_sensors, target_in_the_air=True,
                                grasping=False, joint_control=joint_control, has_obj=False,
                                tip_penalty=tip_penalty, tipping_threshold=tipping_threshold, force_angle_reward_factor=force_angle_reward_factor,
                                noise_stds=noise_stds, target_min_distance=target_min_distance, target_min_distance_xy=target_min_distance_xy,
                                checkReachability=checkReachability)


class KukaSlideEnv(KukaBulletMGEnv):
    def __init__(self, render=True, binary_reward=True, joint_control=False, distance_threshold=0.05,
                 visualize_target=True, gripper_type='parallel_jaw',
                 # unused args, just to make it compatible with the make_env() method
                 image_observation=False, goal_image=False, depth_image=False,
                 camera_setup=None, observation_cam_id=0, goal_cam_id=0):
        KukaBulletMGEnv.__init__(self, render=render, binary_reward=binary_reward, distance_threshold=distance_threshold,
                                 image_observation=False, visualize_target=visualize_target,
                                 gripper_type=gripper_type, obj_range=0.1, target_range=0.2,
                                 table_type='long_table', target_in_the_air=False, end_effector_start_on_table=True,
                                 grasping=False, joint_control=joint_control, has_obj=True)
