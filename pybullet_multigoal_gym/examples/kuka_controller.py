import matplotlib.pyplot as plt

import pybullet_multigoal_gym as pmg
from pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs import \
    KukaTipOverEnv
import time
from seer.controller import ForceAngleController
import numpy as np
from seer.environment import GymEnvironment
from seer.trajectory import TestTrajectory, SimpleTrajectory
import time 
from typing import List
from pybullet_multigoal_gym.utils.noise_generation import add_noise
# f, axarr = plt.subplots(1, 2)
from gym.utils import seeding 
import wandb

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1:
        return v
    if norm == 0: 
       return v
    return v / norm

def get_action(env, newJointPositions, action_noise_std=0.05):
    currentJointPositions = env.robot.joint_state_target
    delta = np.array(newJointPositions) - np.array(currentJointPositions)
    maxComponent = np.max(np.abs(delta))    
    # delta /= (maxComponent + 1e-3)
    if action_noise_std > 0.0:
        delta = add_noise(delta, action_noise_std)
    delta = normalize(delta)
    #     delta = np.clip(delta, -1,1)
    # deltaUnit = normalize(delta)
    action = delta
    return action

def distance(vec1 : List[float], vec2: List[float]):
    assert len(vec1) == len(vec2)
    return np.linalg.norm(np.array(vec1)-np.array(vec2))

def genRandomGoal(robot, seed=11):
    center = robot.end_effector_tip_initial_position.copy()
    np_random, seed = seeding.np_random(seed)
    while True:
        # TODO currently generating points that are potentially to close or too far from the robot arm to reach
        desired_goal = np_random.uniform(robot.target_bound_lower, robot.target_bound_upper)
        if np.linalg.norm(desired_goal - center) > 0.1:
            return desired_goal

def run(env, seed=11):
    obs = env.reset()
    done = False
    env.seed(seed)
    controllerEnv =  GymEnvironment(env)
    controller = ForceAngleController(controllerEnv)
    testTrajectory = TestTrajectory(None, None)
    # testTrajectory = SimpleTrajectory(controller.getEndEffectorWorldPosition(), [0,0,1.5], time.time())
    # testTrajectory = SimpleTrajectory(controller.getEndEffectorWorldPosition(), controller.getEndEffectorWorldPosition(), time.time())
    controller.setTrajectory(testTrajectory)
    currentGoal = [0.,0.,0.,]
    timeOfReset = time.time()
    stepsSinceReset = 0
    successes = 1 
    fails = 1
    tipovers = 0
    steps = 0
    while True:
        stepsSinceReset += 1
        steps += 1
        newJointPositions = controller.getNextJointPositions()
        action = get_action(env, newJointPositions, 0)
        # action = controller.trajectory.getPosition(time.time())

        obs, reward, done, info = env.step(action)
        
        desiredGoal = obs["desired_goal"]
        # desiredGoal = genRandomGoal(env.robot, seed)
        # env.desiredGoal = desiredGoal

        if distance(desiredGoal, currentGoal) > 0.1:
            currentPosition = controller.getEndEffectorWorldPosition()
            trajectory = SimpleTrajectory(currentPosition, desiredGoal, time.time())
            controller.setTrajectory(trajectory)
            currentGoal = desiredGoal

        goalAchieved = info["goal_achieved"]

        if goalAchieved:
            successes += 1
            env.reset()
            if wandb.run:
                wandb.log({"time_to_goal": time.time()-timeOfReset,
                           "steps_to_goal": stepsSinceReset}, step=env.total_steps)
            timeOfReset = time.time()
            stepsSinceReset = 0
        elif time.time() - timeOfReset > 10:
            # if takes longer than 10s assume it cannot reach target
            fails += 1 
            if env.tipped_over():
                tipovers += 1
            env.reset()
            timeOfReset = time.time()
            stepsSinceReset = 0
        
        if wandb.run:
            wandb.log({"success_rate" : successes/(successes + fails),
                       "successes" : successes,
                       "fails" : fails,
                       'tipovers': tipovers,
                       "steps" : steps}, step=env.total_steps)
            

if __name__ == "__main__":
    env: KukaTipOverEnv = pmg.make_env(task='tip_over',
                   gripper='parallel_jaw',
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
                   plane_position=[0,0,-0.58],
                   )
    run(env, seed=11)
