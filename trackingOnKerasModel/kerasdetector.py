'''
Script to test traffic light localization and detection
'''

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt
import time
from glob import glob
cwd = os.path.dirname(os.path.realpath(__file__))

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import tensorflow as tf

# Uncomment the following two lines if need to use the visualization_tunitls
#os.chdir(cwd+'/models')
#from object_detection.utils import visualization_utils as vis_util

# set tf backend to allow memory to grow, instead of claiming everything
class CarDetector(object):
    def __init__(self):

        self.car_boxes = []
        
        os.chdir(cwd)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        # set the modified tf session as backend in keras
        keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


        # adjust this to point to your downloaded/trained model
        self.model_path = os.path.join('.', 'trainedModel', '50.h5')

        # load retinanet model
        self.model = models.load_model(self.model_path, backbone_name='resnet50')

        # load label to names mapping for visualization purposes
        self.labels_to_names = {1: 'car', 2: 'bus', 3: 'van', 4: 'others'}
        
        # setup tensorflow graph
        self.detection_graph = tf.Graph()
    
    # Helper function to convert image into numpy array    
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)       
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):
    
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)       
        
    def get_localization(self, image,visual=False):  
        """Determines the locations of the traffic light in the image

        Args:
            image: camera image

        Returns:
            list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]

        """
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = preprocess_image(bgr)
        image, scale = resize_image(image)
        # preprocess image for network
        boxes, scores, classes = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        
        boxes /= scale
        
        boxes=np.squeeze(boxes)
        classes =np.squeeze(classes)
        scores = np.squeeze(scores)

        cls = classes.tolist()
        
        # The ID for car is 1
        idx_vec = [i for i, v in enumerate(cls) if ((v==1) and (scores[i]>0.3))]
        
        if len(idx_vec) ==0:
            print('no detection!')
        else:
            tmp_car_boxes=[]
            for idx in idx_vec:
                #dim = image.shape[0:2]
                #box = self.box_normal_to_pixel(boxes[idx], dim)
                box = boxes[idx].astype(int)
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                ratio = box_h/(box_w + 0.01)
                
                if ((ratio < 0.8) and (box_h>20) and (box_w>20) and scores[idx] > 0.5):
                    tmp_car_boxes.append(box)
                    print(box, ', confidence: ', scores[idx], 'ratio:', ratio)
                else:
                    print('wrong ratio or wrong size or low confidence, ', box, ', confidence: ', scores[idx], 'ratio:', ratio)

            self.car_boxes = tmp_car_boxes
             
        return self.car_boxes
        
if __name__ == '__main__':
        
        det =CarDetector()
        os.chdir(cwd)
        TEST_IMAGE_PATHS= glob(os.path.join('test_images/', '*.jpg'))
        
        for i, image_path in enumerate(TEST_IMAGE_PATHS[0:2]):
            print('')
            print('*************************************************')
            
            img_full = Image.open(image_path)
            img_full_np = det.load_image_into_numpy_array(img_full)
            img_full_np_copy = np.copy(img_full_np)
            start = time.time()
            b = det.get_localization(img_full_np, visual=False)
            end = time.time()
            print('Localization time: ', end-start)
#            
            
