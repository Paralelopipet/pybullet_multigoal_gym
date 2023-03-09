import numpy as np

def add_noise(action, sigma):
    mean = 0

    if type(action) == list:
        action = np.array(action)

    # generate gaussian noise
    noise = np.random.normal(0.0, sigma, action.shape)
    
    # add noise to actions 
    action_with_noise = np.add(action, noise)
    
    return action_with_noise

def add_noise_recursive(action, sigma):
    mean = 0.0

    if(type(action) == float or type(action) == np.float64):
        return action + np.random.normal(mean, sigma)

    if type(action) == tuple:
        action = list(action)

    return_action = []
    for elem in action:
        rec_return = add_noise_recursive(elem, sigma)
        return_action = (return_action, [rec_return])

    return return_action


def flatten_rec(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, tuple):
            flat_list.extend(flatten_rec(item))
        else:
            flat_list.append(item)
    return flat_list

def add_noise_to_observations(joint_poses, joint_velocities, joint_forces, joint_torques, centre_of_mass, noise_stds):
    measures = [joint_poses, joint_velocities, joint_forces, joint_torques, centre_of_mass]
    noise_stds = [noise_stds.get('pos', 0.0), noise_stds.get('vel', 0.0), noise_stds.get('tor', 0.0), noise_stds.get('tor', 0.0), noise_stds.get('com', 0.0)]
    
    for i, (measure, sigma) in enumerate(zip(measures, noise_stds)):
        measure = flatten_rec(measure)
        measures[i] = add_noise(measure, sigma)
    
    return measures
