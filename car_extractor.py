import tensorflow as tf
import numpy as np
from PIL import Image

from absl import flags, app

from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.utils import draw_outputs, load_darknet_weights
from yolov3_tf2.dataset import transform_images

app.parse_flags_with_usage(['yolo_iou_threshold'])

def get_model(path):
    yolo = YoloV3(classes=80)
    yolo.load_weights(path)
    return yolo

def find_cars(image, yolo):
    #img_arr = np.array(image_orig)
    if type(image) == list:
        images = np.array([np.array(i) for i in image])
        image = transform_images(images, 416)
    else:
        image = tf.expand_dims(np.array(image), axis=0)
        image = transform_images(image, 416)

    boxes, scores, classes, nums = yolo.predict(image, steps = 1)
    
    filtered = []
    for i in range(len(boxes)):
        # If class detected is a car and the 0.95 is to remove the box corresponding to the camera car roof.
        filtered_boxes = [box for box, cla in list(zip(boxes[i], classes[i])) if cla == 2 and box[3] < 0.97] 
        filtered.append(filtered_boxes)
        
    return filtered


def extract_car_images(image, car_boxes, margin=0):
    im_arr = np.array(image)
    rois = []
    for i in range(len(car_boxes)):
        
        point1 = [int(3384 * car_boxes[i][0] - margin), int(2710 * car_boxes[i][1] - margin)]
        point2 = [int(3384 * car_boxes[i][2] + margin), int(2710 * car_boxes[i][3] + margin)]
        
        point1[0] = min(max(point1[0], 0), im_arr.shape[1]-1) #y
        point1[1] = min(max(point1[1], 0), im_arr.shape[0]-1) #x
        point2[0] = min(max(point2[0], 0), im_arr.shape[1]-1) #y
        point2[1] = min(max(point2[1], 0), im_arr.shape[0]-1) #x
        
        try:
            rois.append(
                Image.fromarray(
                    im_arr[point1[1]:point2[1], 
                           point1[0]:point2[0]]
                )
            )
        except:
            print(point1, point2)
            print(im_arr.shape)
            display(draw_car_boxes(image, car_boxes))
            pass
    return rois