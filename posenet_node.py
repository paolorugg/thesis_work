import rclpy 
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # 
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2
from pycoral.utils import edgetpu                      #
from tflite_runtime.interpreter import load_delegate   #
from tflite_runtime.interpreter import Interpreter     # For the use of the coral gpu and PoseNet
import enum
import math
import os
import platform
import sys
import time
import argparse
import collections
from functools import partial
import re
import time
import numpy as np
from PIL import Image as PIL_image

from tesi.pose_engine_mod import *
from tesi.sort import *

# Initialization for the SORT algorithm
args_sort = parse_args_sort()
mot_tracker = Sort(max_age=args_sort.max_age, 
                   min_hits=args_sort.min_hits,
                   iou_threshold=args_sort.iou_threshold)

# Start-up of the posenet model
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
parser.add_argument('--model', help='.tflite model path.', required=False)
parser.add_argument('--res', help='Resolution', default='640x480',
                    choices=['480x360', '640x480', '1280x720'])
parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
parser.add_argument('--jpeg', help='Use image/jpeg input', action='store_true')
args = parser.parse_args()

default_model = '/home/paolo/project-posenet/models/mobilenet/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite' # Path of mobilenet model, downloaded from coral.ai

if args.res == '480x360':
    src_size = (640, 480)
    appsink_size = (480, 360)
    model = args.model or default_model % (353, 481)
elif args.res == '640x480':
    src_size = (640, 480)
    appsink_size = (640, 480)
    model = args.model or default_model % (481, 641)
elif args.res == '1280x720':
    src_size = (1280, 720)
    appsink_size = (1280, 720)
    model = args.model or default_model % (721, 1281)
    
# Definition of the possible links between detected keypoints   
EDGES = (
    (KeypointType.NOSE, KeypointType.LEFT_EYE),
    (KeypointType.NOSE, KeypointType.RIGHT_EYE),
    (KeypointType.NOSE, KeypointType.LEFT_EAR),
    (KeypointType.NOSE, KeypointType.RIGHT_EAR),
    (KeypointType.LEFT_EAR, KeypointType.LEFT_EYE),
    (KeypointType.RIGHT_EAR, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
    (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
    (KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
    (KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
    (KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
    (KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE),
)

class PoseNet_Node(Node):
    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('node')
        self.engine = PoseEngine(model)

        # Creation of publishers
        self.publisher_1 = self.create_publisher(Image, '/frames_with_pose', 10) 
        self.publisher_2 = self.create_publisher(Float32MultiArray, '/distance_with_depth_camera', 10) 
    
        # Creation of the subscribers
        self.subscription1 = self.create_subscription(Image, '/camera/color/image_raw', self.listener_callback, 10)
        self.subscription2 = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.acquire_depth_image, 10)
        self.subscription1 # Prevent unused variable warning
        self.subscription2
       
        self.br = CvBridge() # Used to convert between ROS and OpenCV images
        self.depth_image = np.zeros((480,640))
        
    def draw_pose(self,img, pose, threshold=0.2): 
        xys = {}
        for label, keypoint in pose.keypoints.items(): # Print recognized keypoints on image
            if keypoint.score < threshold: continue
            kp_x = int(keypoint.point[0])
            kp_y = int(keypoint.point[1])

            xys[label] = (kp_x, kp_y)
            img = cv2.circle(img, (kp_x, kp_y), 5, (255, 153, 51), -1)
            
        for a, b in EDGES:
            if a not in xys or b not in xys: continue
            ax, ay = xys[a] 
            bx, by = xys[b]
            img = cv2.line(img, (ax, ay), (bx, by), (51, 255, 255), 2) 
            # Yellow edges, cyan keypoint
        
    
    def put_poses_upon_image(self, img, poses):
        for pose in poses:
               self.draw_pose(img, pose)
        return img
    
    def get_x_distance(self, x1, x2, z):
        px_distance= x2 - x1  # Positive -> point to the right of the center
        x_distance= px_distance * z / 617.4224243164062  # Pixel distance * z /fx
        return x_distance
    
    def medium_point_distance(self, x1, y1, x2, y2, img, pred_id):
        x_med = (x1+x2)/2
        y_med = (y1+y2)/2
        
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        img = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2) # Print red box
        
        pred_id_int=int(pred_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (int(x1+3), int(y2-3))
        fontScale = 0.8
        img = cv2.putText(img, str(pred_id_int), org, font, fontScale, (0, 0, 255), 2, cv2.LINE_AA)  # Print id
               
        z_distance = self.depth_image[int(round(y_med)), int(round(x_med))] / 1000 # Distance in meters
        x_distance = self.get_x_distance(316.72235107421875, x_med, z_distance) # 316.7 is center point x coordinate
        return z_distance, x_distance

    

    def get_person_distance(self, tracker, img, seq_id):
        return self.medium_point_distance(tracker[0], tracker[1], tracker[2], tracker[3], img, seq_id)
    
    
    
    def listener_callback(self, data):
        # Convert ROS Image message to OpenCV image
        frame = self.br.imgmsg_to_cv2(data)
        frame = cv2.resize(frame, (src_size[0], src_size[1])) 

        frame_rgb = frame[:, :, ::-1].copy()   # Switch from bgr to rgb encoding 
    
        poses_result, inference_time = self.engine.DetectPosesInImage(frame_rgb) # Run inference
        if poses_result:            
            ## SORT part
            dets_list = []
            boxes = []
            trackers= np.empty((0,5))
            for pose in poses_result:
                x_min = min(pose[0][5].point[0], pose[0][6].point[0])  - 20
                y_min = min(pose[0][5].point[1], pose[0][6].point[1])  - 20
                x_max = max(pose[0][5].point[0], pose[0][6].point[0])  + 20
                y_max = max(pose[0][5].point[1], pose[0][6].point[1])  + 20
                boxes.append([x_min, y_min, x_max, y_max, pose[1]]) # Minimum and maximum are done between shoulders coordinates
            if len(boxes)>0:
                dets = np.array(boxes)
                trackers = mot_tracker.update(dets)

            for person in range(trackers.shape[0]):
                person_id = trackers[person][4]
                z_distance, x_distance = self.get_person_distance(trackers[person,:], frame_rgb, person_id)
                msg = Float32MultiArray()
                msg.data = [float(person_id), float(z_distance), float(x_distance)] 
                self.publisher_2.publish(msg)
            
            final_output = self.put_poses_upon_image(frame_rgb, poses_result)
            
        else:
            final_output = frame_rgb # If PoseNet detects no pose, the simple webcam is shown
        
        self.publisher_1.publish(self.br.cv2_to_imgmsg(final_output))
        final_output = cv2.resize(final_output, (1100, 780))  # Eventual resize
                
        cv2.imshow("PoseNet", final_output)
        cv2.waitKey(1)
        
   
    def acquire_depth_image(self, data):
        self.depth_image = self.br.imgmsg_to_cv2(data)


    
        
        
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  node = PoseNet_Node()
  
  # Spin the node so the callback function is called.
  rclpy.spin(node)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  node.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()