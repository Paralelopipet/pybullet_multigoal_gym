import numpy as np

def gravity_vector(angle):
    # Generate a random angle between 0 and 2*pi
    theta = np.random.uniform(0, 2*np.pi)
    # Generate a random angle between 0 and the given angle
    phi = np.random.uniform(0, angle)
    # Calculate the x, y, and z components of the vector
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z]) * (-9.81)