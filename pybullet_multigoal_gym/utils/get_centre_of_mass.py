import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

def get_centre_of_mass(pybullet, robot_id, total_mass: float) -> NDArray:
    # Calculate the cener of mass of the robot_id
    global_center_of_mass = np.array([0, 0, 0], dtype=np.float32)
    for link_idx in range(pybullet.getNumJoints(robot_id)):
        link_mass = pybullet.getDynamicsInfo(robot_id, link_idx)[0]
        link_center_of_mass = np.array(pybullet.getLinkState(robot_id, link_idx)[0])
        global_center_of_mass += (link_mass / total_mass) * link_center_of_mass
    return _transform_to_base_coordinate_system(pybullet, global_center_of_mass, robot_id)

def _transform_to_base_coordinate_system(pybullet, global_center_of_mass: NDArray, robot_id) -> NDArray:
    position, orientation = pybullet.getBasePositionAndOrientation(robot_id)
    return Rotation.from_quat(orientation).apply(
        global_center_of_mass - np.array(position, dtype=np.float32)
    )
