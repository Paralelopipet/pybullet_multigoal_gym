from collections import defaultdict
import os
from typing import List

import numpy as np
from numpy.typing import NDArray
from pybullet_multigoal_gym.utils.get_centre_of_mass import get_centre_of_mass
from pybullet_multigoal_gym.utils.gravity import gravity_vector
from pybullet_multigoal_gym.utils.noise_generation import add_noise_to_observations
from scipy.spatial.transform import Rotation

from pybullet_multigoal_gym.envs.base_envs.base_env import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka_with_box import KukaBox
from pybullet_multigoal_gym.utils.assets_dir import CUBE_LINK_NAME
from seer.stability_metrics.adapter.stability_metric_adapter import \
    StabilityMetricAdapter
from seer.stability_metrics.adapter.types import RobotConfig, RobotState
import wandb


class KukaBullet3Env(BaseBulletMGEnv):
    """
    Base class for the OpenAI multi-goal manipulation tasks with a Kuka iiwa 14 robot
    """

    def __init__(self, render=True, binary_reward=True, gravity_angle = 0.0,
                 image_observation=False, goal_image=False, depth_image=False, visualize_target=True,
                 camera_setup=None, observation_cam_id=None, goal_cam_id=0,
                 gripper_type='parallel_jaw', obj_range=0.15, target_range=0.15, plane_position = [0.,0.,-1.], has_spring=False,  joint_force_sensors=False,
                 target_in_the_air=True, end_effector_start_on_table=False,
                 distance_threshold=0.05, joint_control=True, grasping=False, has_obj=False, tip_penalty=-100.0, tipping_threshold=0.5,
                 force_angle_reward_factor=1.0, noise_stds = {}, target_min_distance=0.1, target_min_distance_xy=0.1, checkReachability=True):
        if observation_cam_id is None:
            observation_cam_id = [0]
        self.binary_reward = binary_reward
        self.gravity_angle = gravity_angle
        self.image_observation = image_observation
        self.goal_image = goal_image
        if depth_image:
            self.render_mode = 'rgbd_array'
        else:
            self.render_mode = 'rgb_array'
        self.visualize_target = visualize_target
        self.observation_cam_id = observation_cam_id
        self.goal_cam_id = goal_cam_id

        self.target_in_the_air = target_in_the_air
        self.distance_threshold = distance_threshold
        self.joint_control = joint_control
        self.grasping = grasping
        self.has_obj = has_obj
        self.obj_range = obj_range
        self.target_range = target_range
        self.target_min_distance = target_min_distance
        self.target_min_distance_xy = target_min_distance_xy
        self.checkReachability = checkReachability
        self.plane_position = plane_position
        self.has_spring = has_spring
        self.joint_force_sensors =  joint_force_sensors
        self.object_assets_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "objects")
        self.objects_urdf_loaded = False
        self.object_bodies = {
            'block': None,
            'target': None
        }
        self.object_initial_pos = {
            'block': [-0.52, 0.0, 0.175, 0.0, 0.0, 0.0, 1.0],
            'target': [-0.52, 0.0, 0.186, 0.0, 0.0, 0.0, 1.0]
        }

        self.tip_penalty = tip_penalty
        self.tipping_threshold = tipping_threshold
        self.force_angle_reward_factor = force_angle_reward_factor

        self.noise_stds = noise_stds# has 'pos', 'vel', 'tor' and 'com'

        self.desired_goal = None
        self.desired_goal_image = None

        self.episode_steps = 0
        self.total_steps = 0
        self.tip_over_count = 0

        # define parameters for work
        self.vel_work_integral = 0 
        self.pos_work_integral = 0 
        self.last_joint_poses = None

        robot = KukaBox (grasping=grasping,
                     joint_control=joint_control,
                     gripper_type=gripper_type,
                     end_effector_start_on_table=end_effector_start_on_table,
                     obj_range=self.obj_range, target_range=self.target_range, plane_position=self.plane_position, has_spring=self.has_spring,  joint_force_sensors=self.joint_force_sensors)

        self._force_angle_calculator = StabilityMetricAdapter.instance(
            RobotConfig(cube_link_name=CUBE_LINK_NAME, urdf_path=robot.model_urdf)
        )
        BaseBulletMGEnv.__init__(self, robot=robot, render=render,
                                 image_observation=image_observation, goal_image=goal_image,
                                 camera_setup=camera_setup,
                                 seed=0, timestep=0.002, frame_skip=20)

    def _log_before_reset(self):
        is_tipped = self.tipped_over() # calculate if angle from z axis is higher than threshold
        self.tip_over_count += float(is_tipped)
        if self.desired_goal is None:
            self.desired_goal = [0,0,0]
        if wandb.run:
            wandb.log({
                'tipped_over': float(is_tipped),
                'tip_over_count': self.tip_over_count,
                'desired_goal_x' : self.desired_goal[0],
                'desired_goal_y' : self.desired_goal[1],
                'desired_goal_z' : self.desired_goal[2],
                'positional work' : self.pos_work_integral,
                'velocity work' : self.vel_work_integral,
                'total work per episode - vel': np.sum(self.vel_work_integral),
                'total work per episode - pos': np.sum(self.pos_work_integral)
        }, step=self.total_steps)


    def _task_reset(self, test=False):
        if not self.objects_urdf_loaded:
            # don't reload object urdf
            self.objects_urdf_loaded = True
            self.object_bodies['target'] = self._p.loadURDF(
                os.path.join(self.object_assets_path, "target.urdf"),
                basePosition=self.object_initial_pos['target'][:3],
                baseOrientation=self.object_initial_pos['target'][3:])
            if not self.visualize_target:
                self.set_object_pose(self.object_bodies['target'],
                                     [0.0, 0.0, -3.0],
                                     self.object_initial_pos['target'][3:])

        # randomize object positions
        object_xyz_1 = None
        if self.has_obj:
            end_effector_tip_initial_position = self.robot.end_effector_tip_initial_position.copy()
            object_xy_1 = end_effector_tip_initial_position[:2]
            while np.linalg.norm(object_xy_1 - end_effector_tip_initial_position[:2]) < 0.1:
                object_xy_1 = self.np_random.uniform(self.robot.object_bound_lower[:-1],
                                                     self.robot.object_bound_upper[:-1])

            object_xyz_1 = np.append(object_xy_1, self.object_initial_pos['block'][2])
            self.set_object_pose(self.object_bodies['block'],
                                 object_xyz_1,
                                 self.object_initial_pos['block'][3:])

        # generate goals & images
        self._generate_goal(current_obj_pos=object_xyz_1, minimum_distance=self.target_min_distance, minimum_distance_xy=self.target_min_distance_xy)
        if self.goal_image:
            self._generate_goal_image(current_obj_pos=object_xyz_1)
        
        #reset gravity
        self.gravity_vec, self.gravity_phi, self.gravity_theta = gravity_vector(self.gravity_angle)
        self._p.setGravity(self.gravity_vec[0], self.gravity_vec[1], self.gravity_vec[2])

        # reset time
        self.episode_steps = 0

        # reset work integrals
        self.last_joint_poses = None
        self.pos_work_integral = 0
        self.vel_work_integral = 0

    def _generate_goal(self, minimum_distance, minimum_distance_xy, current_obj_pos=None):
        if current_obj_pos is None:
            # generate a goal around the gripper if no object is involved
            center = self.robot.end_effector_tip_initial_position.copy()
        else:
            center = current_obj_pos

        # generate the 3DoF goal within a 3D bounding box such that,
        #       it is at least 0.02m away from the gripper or the object
        while True:
            self.desired_goal = self.np_random.uniform(self.robot.target_bound_lower,
                                                       self.robot.target_bound_upper)
            # NOTE For testing goal distances, override the goal here
            # self.desired_goal = np.array([-0.1179, -0.08029, 0.2271])
            # self.desired_goal = np.array([-0.9768, 0.3392, 0.7156])
            distance = np.linalg.norm(self.desired_goal - center)
            xy_distance = np.sqrt(np.sum(self.desired_goal[:2] ** 2))
            if distance > minimum_distance and xy_distance > minimum_distance_xy:
                if self.checkReachability:
                    if self.robot.is_point_reachable(self.desired_goal):
                        break
                else:
                    print("accepted goal distance:", distance, 'xy distance:', xy_distance, 'minimums', minimum_distance, minimum_distance_xy)
                    break

        if not self.target_in_the_air:
            self.desired_goal[2] = self.object_initial_pos['block'][2]
        elif self.grasping:
            # with .5 probability, set the pick-and-place target on the table
            if self.np_random.uniform(0, 1) >= 0.5:
                self.desired_goal[2] = self.object_initial_pos['block'][2]

        if self.visualize_target:
            self.set_object_pose(self.object_bodies['target'],
                                 self.desired_goal,
                                 self.object_initial_pos['target'][3:])
        # self.desired_joint_goal = np.array(self.robot.compute_ik(self.desired_goal))


    def _step_callback(self):
        pass

    def _get_obs(self):
        # robot state contains gripper xyz coordinates, orientation (and finger width)
        gripper_xyz, gripper_rpy, gripper_finger_closeness, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_vel, joint_poses, joint_velocities, joint_forces, joint_torques = self.robot.calc_robot_state()
        assert self.desired_goal is not None
        state = gripper_xyz
        achieved_goal = gripper_xyz.copy()
        assert not self.grasping, "grasping should not be true when there is no objects"

        centre_of_mass = self.get_centre_of_mass()

        # calculate the work done! BOI
        if self.last_joint_poses is None:
            self.last_joint_poses = np.array(joint_poses)

        # positional work done
        pos_work = np.add(np.array(joint_poses), -self.last_joint_poses) * joint_torques
        vel_work = self.dt * np.multiply((np.array(joint_velocities)), joint_torques)

        self.pos_work_integral += np.clip(pos_work, 0.,np.inf)
        self.vel_work_integral += np.clip(vel_work, 0.,np.inf)


        [joint_poses, joint_velocities, joint_forces, joint_torques, centre_of_mass] = add_noise_to_observations(joint_poses, joint_velocities, joint_forces, joint_torques, centre_of_mass, self.noise_stds)

        if self.joint_control:
            state = np.concatenate((joint_poses, gripper_xyz, centre_of_mass, joint_velocities, joint_forces, joint_torques, state))

        # count time
        self.episode_steps += 1
        self.total_steps += 1
        
        GRAVITY_FACTOR = np.pi * 200
        state = np.concatenate((state, [self.gravity_phi / GRAVITY_FACTOR, self.gravity_theta / GRAVITY_FACTOR]))
        state = np.concatenate((state, [self.episode_steps]))
        if wandb.run:
            wandb.log({
                'force_angle': self.force_angle(centre_of_mass),
                'episode_steps': self.episode_steps,
                'max_joint_vel' : max(joint_velocities),
                'joint_velocities' : joint_velocities,
                'joint_torques' : joint_torques,
                'simulation_time' : self.dt*self.total_steps # real world time that has passed within the simulation
        }, step=self.total_steps)

        obs_dict = {'observation': state.copy(),
                    "policy_state": state.copy(),
                    'achieved_goal': achieved_goal.copy(),
                    'desired_goal': self.desired_goal.copy(),
                    # 'desired_joint_goal': self.desired_joint_goal.copy(),
                    }
        return obs_dict

    def _compute_reward(self, achieved_goal, desired_goal, observation):
        assert achieved_goal.shape == desired_goal.shape
        reward = 0
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        not_achieved = (d > self.distance_threshold)
        if self.binary_reward:
            return -not_achieved.astype(np.float32), ~not_achieved
        

        achieved_reward = 100.0 # TODO Move to param
        distance_factor = 10.0 # TODO Move to param
        if not not_achieved:
            reward += achieved_reward
        reward += -d * distance_factor
        joints = observation[:7]
        gripper_xyz = observation[7:10]
        com = observation[10:13]
        force_angle = self.force_angle(com)
        is_tipped = self._tipped_over(force_angle)

        # Add penalty for bad force angle
        if force_angle == np.inf:
            reward += self.force_angle_reward_factor * 1 # max reward for good stability
        elif force_angle == -np.inf:
            reward += self.tip_penalty
        else:
            if force_angle > 0:
                # reward += self.force_angle_penalty_factor * -1 / (100 * force_angle) + 0.5
                reward += self.force_angle_reward_factor * 1/(1 + np.exp(-force_angle)) - 0.5
            else:
                reward += self.tip_penalty
        # if is_tipped:
        #     reward += self.tip_penalty
        return reward, ~not_achieved

    def set_object_pose(self, body_id, position, orientation=None):
        if orientation is None:
            orientation = self.object_initial_pos['table'][3:]
        self._p.resetBasePositionAndOrientation(body_id, position, orientation)


    def force_angle(self, centre_of_mass: NDArray) -> float:
        # note that all values in the robot state need to be relative to the cube origin!
        # i.e., rotate and translate the forces and moments and centre of mass by the current orientation/position of the cube origin
        fx, fy, fz, mx, my, mz = self._p.getJointState(self.robot.robot_id, self.robot.body_joint_index)[2]
        return self._force_angle_calculator.get(RobotState(
            centre_of_mass=centre_of_mass,
            net_force=np.array([fx,fy,fz]),
            net_moment=np.array([mx,my,mz])
        ))

    def _tipped_over(self, force_angle: float) -> bool:
        return force_angle < 0

    def get_centre_of_mass(self) -> NDArray:
        return get_centre_of_mass(self._p, self.robot.robot_id, self.robot.total_mass)

    def tipped_over(self):
        pos, orientation = self._p.getBasePositionAndOrientation(self.robot.robot_id)
        return self._p.getAxisAngleFromQuaternion(orientation)[-1] > self.tipping_threshold
    
    def simulation_time(self):
        return self.dt*self.total_steps
    
    @property 
    def p(self):
        return self._p 