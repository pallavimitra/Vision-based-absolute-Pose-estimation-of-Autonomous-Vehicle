import numpy as np
import math

CAMERA_fx = 2304.5479
CAMERA_fy = 2305.8757
CAMERA_cx = 1686.2379
CAMERA_cy = 1354.9849

CAMERA_FOV = (CAMERA_cx / CAMERA_fx)
RADS_PIXEL_X = CAMERA_FOV / 3384

IMAGE_WIDTH = 3384
IMAGE_HEIGHT = 2710

CAMERA_MATRIX = np.array([
    [CAMERA_fx,   0,        0],
    [0,        CAMERA_fy,   0],
    [0,           0,        1],
])

def to_cam_xy(world_coords):
    """ Converts world coordinates (X, Y, Z) to the projection on the images (x, y)"""
    if type(world_coords) == list:
        p = np.array(world_coords)
    else:
        p = world_coords
    im_point = np.dot(p, CAMERA_MATRIX)
    im_point[:,0] /= p[:,2]
    im_point[:,1] /= p[:,2]
    
    im_point[:,0] += CAMERA_cx
    im_point[:,1] += CAMERA_cy
    
    return im_point

def to_2pi_angle(radian):
    """Takes a +-π radian and transforms in [0, 2π) range"""
    if radian < 0:
        radian = (2*math.pi) + radian
    return radian

def get_ray_angle(x_pixel, pi_angle=True):
    """ Gets the angle of the ray given a X_PIXEL. 
    If pi_angle = True, then, the angle is expresed in [0,2π) range, else 
    the angle will be expresed in +-rotation from angle 0 (the center of the photo).
    """
    angle = (x_pixel *  RADS_PIXEL_X) - (CAMERA_FOV/2)
    if pi_angle:
        angle = to_2pi_angle(angle)
    return angle

def get_local_rot(ray_angle, global_angle):
    return global_angle - ray_angle

def get_global_angle(ray_angle, local_angle):
    return ray_angle + local_angle
        