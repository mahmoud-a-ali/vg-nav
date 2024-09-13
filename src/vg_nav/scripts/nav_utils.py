import numpy as np
from tf.transformations import quaternion_from_euler


def convert_spherical_to_cartesian(theta, alpha, dist):
    x = np.array(dist * np.sin(alpha) * np.cos(theta), dtype='float32')
    y = np.array(dist * np.sin(alpha) * np.sin(theta), dtype='float32')
    z = np.array(dist * np.cos(alpha), dtype='float32')
    return x, y, z


def convert_cartesian_to_spherical(x, y, z):
    dist = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    alpha = np.arccos(z / dist)
    return theta, alpha, dist


def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return arr / (max_val)


def create_quaternion(yaw):
    return quaternion_from_euler(0, 0, yaw)
