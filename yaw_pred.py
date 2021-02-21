import math
import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)

NUM_BINS = 8
IMAGE_INP_SIZE = 64
DENORM_OFF = 0.7853819264708478


def to_angle(radian):
    """Takes a +-π radian and transforms in [0, 2π) range"""
    if radian < 0:
        radian = (2*math.pi) + radian
    return radian

def to_rotation(radian):
    """Takes an angle and transforms it in +-π range """
    base_angle = 0
    
    if math.sin(radian) < 0:
        base_angle = -(2 * math.pi - radian)
    else:
        base_angle = radian - base_angle
    
    return base_angle

def get_local_rot(ray_angle, global_angle):
    return global_angle - ray_angle

def get_global_rot(ray_angle, local_angle):
    return ray_angle + local_angle

def get_bin(angle):
    """Gets bin nb and offset from that number.
    params: 
        - angle: Angle in radians [0, 2π)
    """
    bin_size = 360 / NUM_BINS
    total_bins = 360//bin_size
    
    degrees = math.degrees(angle) + bin_size/2  #Shift the bins
    bin_number = (degrees // bin_size) % total_bins
    offset = (degrees - (bin_number*bin_size))
    
    if degrees > 360:  #Correct offset if in last semi bin (8 == 0)
        offset = degrees - ((total_bins) * bin_size)
    
    offset = math.radians(offset)

    return bin_number, offset

def prediction_to_yaw(bin_nb, offset, ray_angle):
    """ Takes bin + offset and using the ray angle 
    returns the global rotation of the car """
    bin_size = 2*math.pi / NUM_BINS
    
    # Local rotation of the car in [0, 2π)
    angle = bin_nb * bin_size + offset - bin_size/2  # shift bins
    
    # Global rotation of the car (taking into account the camera ray angle)
    angle = get_global_rot(ray_angle, angle)
    
    if angle < 0:
        angle = 2*math.pi + angle
    
    # Represent the angle as a rotation of [0, π] or [0, -π)
    #angle = to_rotation(angle)
    
    return angle

def angle_distance(angle1, angle2):
    """Returns the shortest distance in degrees between two angles.
    Parameters:
        - angle1, angle2: (Degrees)
    """
    diff = ( angle2 - angle1 + 180 ) % 360 - 180;
    diff =  diff + 360  if diff < -180 else diff
    return diff

def predict_yaw(images, ray_angle, model, return_confidences=False):
    """ Uses a model to return global orientation of a car
    Parameters:
        - image: numpy array of a crop
        - ray_angle: Ray angles expressed in radians
        - model: Multibin prediction model
    """
    
    imgs = []
    for ima in images:
        imgs.append(tf.image.rgb_to_grayscale(tf.image.resize(ima, [IMAGE_INP_SIZE, IMAGE_INP_SIZE]).numpy()) / 255)
    
    p = model.predict(np.array(imgs))#, verbose=True)
    
    p_off = p[1] * DENORM_OFF
    p_bin = p[0]
    
    angles = []
    for i in range(len(p_bin)):
        bin_nb = np.argmax(p_bin[i])
        offset = p_off[i]
    
        ang = prediction_to_yaw(bin_nb, offset, ray_angle[i])[0]
        angles.append(to_rotation(ang))
    
    if return_confidences:
        angles = angles, np.max(p_bin, axis=1)
    return angles

def get_model(path):
    return tf.keras.models.load_model(path)