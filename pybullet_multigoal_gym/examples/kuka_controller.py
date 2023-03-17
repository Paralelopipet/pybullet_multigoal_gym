import matplotlib.pyplot as plt

import pybullet_multigoal_gym as pmg
from pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs import \
    KukaTipOverEnv
import time
from seer.controller import ForceAngleController
import numpy as np
from seer.environment import GymEnvironment
from seer.trajectory import TestTrajectory, SimpleTrajectory, SimpleTrajectory2
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

def run(env, *, seed: int, num_epochs: int, num_cycles: int, num_episodes: int, action_noise_std):
    obs = env.reset()
    done = False
    env.seed(seed)
    controllerEnv =  GymEnvironment(env)
    controller = ForceAngleController(controllerEnv)
    testTrajectory = TestTrajectory(None, None)
    # testTrajectory = SimpleTrajectory(controller.getEndEffectorWorldPosition(), [0,0,1.5], time.time())
    # testTrajectory = SimpleTrajectory(controller.getEndEffectorWorldPosition(), controller.getEndEffectorWorldPosition(), time.time())
    #
    controller.setTrajectory(testTrajectory)
    currentGoal = [0.,0.,0.,]
    # timeOfReset = time.time()
    timeOfReset = env.simulation_time() + 5
    stepsSinceReset = 0
    successes = 1 
    fails = 1
    steps = 0
    maxEpisodeSteps = 100
    position_std = env.noise_stds['pos']
    centerOfMass_std = env.noise_stds['com']
    goalReachedYet = False
    for epoch in range(num_epochs):
        for cycle in range(num_cycles):
            print("Epoch", epoch, "Cycle", cycle)
            for episode in range(num_episodes):
                episodeFinished = False
                while not episodeFinished:
                    stepsSinceReset += 1
                    steps += 1
                    newJointPositions = controller.getNextJointPositions(env.simulation_time())
                    action = get_action(env, newJointPositions, 0)
                    # action = controller.trajectory.getPosition(time.time())
                    # add noise to action
                    if action_noise_std>0.0:
                        action = add_noise(action, action_noise_std)
                        action = np.clip(action, -1.0, 1.0)
                        
                    obs, reward, done, info = env.step(action)
                    
                    desiredGoal = obs["desired_goal"]
                    currentPosition = controller.getEndEffectorWorldPosition()
                    # print(distance(currentPosition, desiredGoal))
                    # desiredGoal = genRandomGoal(env.robot, seed)
                    # env.desiredGoal = desiredGoal

                    if distance(desiredGoal, currentGoal) > 0.1:
                        currentPosition = controller.getEndEffectorWorldPosition()
                        trajectory = SimpleTrajectory2(currentPosition, desiredGoal, env.simulation_time())
                        controller.setTrajectory(trajectory)
                        currentGoal = desiredGoal

                    tipoverResponsesInitiated = controller.tipoverResponsesSinceLastCheck()

                    goalAchieved = info["goal_achieved"]

                    if goalReachedYet:
                        if stepsSinceReset > maxEpisodeSteps:
                            successes += 1 
                            env.reset()
                            timeOfReset = env.simulation_time()
                            stepsSinceReset =0 
                            goalReachedYet = False
                            episodeFinished = True
                        else:
                            pass 
                    else:
                        if goalAchieved:
                            goalReachedYet = True 
                            if wandb.run:
                                wandb.log({"time_to_goal": env.simulation_time()-timeOfReset,
                                        "steps_to_goal": stepsSinceReset}, step=env.total_steps)
                        elif stepsSinceReset > maxEpisodeSteps:
                            fails += 1
                            env.reset()
                            timeOfReset = env.simulation_time()
                            stepsSinceReset = 0
                            goalReachedYet = False
                            episodeFinished = True

                    
                    if wandb.run:
                        wandb.log({"successful_episodes_rate" : successes/(successes + fails),
                                "successful_episodes_count" : successes,
                                "failed_episodes_count" : fails,
                                "tipover_responses" : tipoverResponsesInitiated}, step=env.total_steps)
