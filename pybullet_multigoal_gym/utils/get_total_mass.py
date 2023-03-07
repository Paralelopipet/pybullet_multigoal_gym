def get_total_mass(pybullet, robot_id) -> float:
    total_mass = 0
    for link_idx in range(pybullet.getNumJoints(robot_id)):
        link_mass = pybullet.getDynamicsInfo(robot_id, link_idx)[0]
        total_mass += link_mass
    return total_mass
