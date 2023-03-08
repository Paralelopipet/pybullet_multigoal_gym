import numpy as np

def add_noise(action, sigma):
    mean = 0

    # generate gaussian noise
    noise = np.random.normal(0.0, sigma, action.shape)
    
    # add noise to actions 
    action_with_noise = np.add(action, noise)
    
    return action_with_noise

def add_noise_to_observations(obs : dict, sigma) -> dict:
    for key in obs:
        val = obs[key]
        
        # generate gaussian noise
        noise = np.random.normal(0.0, sigma, val.shape)

        obs[key] = np.add(val, noise)
    
    return obs
