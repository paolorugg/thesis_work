import rclpy 
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 
import numpy as np
from dt_apriltags import Detector
import math
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R


camera_params = [617.4224243164062, 617.7899780273438, 316.72235107421875, 244.21875] #fx, fy, xc, yc
tag_size = 0.172 # Real life dimension of the tag side in meters
tag_x_y = [1.45, 0.5] # Absolute position of the tag
angle_rot = -60 # Default angle of the tag wrt x axis of the map. Y axis of the tag is pointing down
angle_to_be_used = (90 - angle_rot) * math.pi / 180 # In radians, calculations involve 90-alpha
tag_z_parallel = -1 # x axis of global map wrt to tag z axis: -1 = antiparallel, 1 = parallel  

class AprilTag_Node(Node):
    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('node')
    
        # Create the subscriber
        self.subscription1 = self.create_subscription(Image, '/camera/color/image_raw', self.listener_callback, 1)
        self.subscription1 # Prevent unused variable warning
        self.publisher_ = self.create_publisher(Point, '/pose_relocalization', 10) # Topic that permits to overwrite odometry
        self.at_detector = Detector(searchpath=['apriltags'],
                              families='tag36h11',
                              nthreads=1,
                              quad_decimate=1.0,
                              quad_sigma=0.0,
                              refine_edges=1,
                              decode_sharpening=0.25,
                              debug=0)
       
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()
        xyz= Point()
        xyz.x= 0.0
        xyz.y= 0.0 
        xyz.z= 0.0
        self.publisher_.publish(xyz) # Shifts base_footprint to global point (0,0)

        
    def listener_callback(self, data):
        # Convert ROS Image message to OpenCV image
        frame = self.br.imgmsg_to_cv2(data)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tags = self.at_detector.detect(gray_frame, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

        frame_rgb = frame[:, :, ::-1].copy() 
        
        if tags != []:
          v1 = (int(tags[0].corners[0][0]), int(tags[0].corners[0][1])) # bottom-left
          v2 = (int(tags[0].corners[1][0]), int(tags[0].corners[1][1])) # bottom-right
          v3 = (int(tags[0].corners[2][0]), int(tags[0].corners[2][1])) # top-right
          v4 = (int(tags[0].corners[3][0]), int(tags[0].corners[3][1])) # top-left
          pts = np.array([v1,v2,v3,v4], np.int32)
          pts = pts.reshape((-1,1,2))
          center_x = int(tags[0].center[0])
          center_y = int(tags[0].center[1])

          frame_rgb = cv2.polylines(frame_rgb,[pts],True,(0,0,255), 2) # Plots a red polygon around the tag
          frame_rgb = cv2.circle(frame_rgb, (center_x, center_y), 3, (0, 0, 255), -1) # Plots the center
          
          if (v1[0]>10 and v4[0]>10 and v2[0]<630 and v3[0]<630 and v1[1]<475 and v2[1]<475 and v3[1]>10 and v4[1]>10): # Vertices of the tag shall not be at the borders
            pose_R = tags[0].pose_R
            pose_t = np.array([tags[0].pose_t[0][0],tags[0].pose_t[1][0], tags[0].pose_t[2][0]])
            x = pose_t[0]
            y = pose_t[1]
            z = pose_t[2]
            estimated_rot = R.from_matrix(pose_R)
            pitch_deg = (estimated_rot.as_euler('xyz', degrees=True))[1]
            pitch = pitch_deg * math.pi / 180
            OH = z * math.cos(pitch)
            HM = x * math.sin(pitch)
            FK = x * math.cos(pitch)
            FH = z * math.sin(pitch)
            x_distance_from_marker = OH + HM + 0.1 * math.cos(pitch)  #0.1 meters is the distance camera-center of the robot
            y_distance_from_marker = FH + 0.1 * math.sin(pitch) - FK 
            x_w = abs(x_distance_from_marker * math.sin(angle_to_be_used) + y_distance_from_marker * math.cos(angle_to_be_used))
            y_w = - x_distance_from_marker * math.cos(angle_to_be_used) + y_distance_from_marker * math.sin(angle_to_be_used)
            if (x_w < 2.5 and abs(y_w) < 2.5): # If the tag is distant is ignored, as estimation is less accurate
              xyz= Point()
              xyz.x= rotated_tag_4[0] + tag_z_parallel * x_w  # Robot x position in global map [meters]
              xyz.y= rotated_tag_4[1] + tag_z_parallel * y_w # Robot y position in global map [meters]
              xyz.z= (pitch_deg - angle_rot + 180 * ((tag_z_parallel-1)/2 +1)) * math.pi / 180   # Angle (radians) of rotation of the robot
              self.publisher_.publish(xyz)

              ## Print the pitch value on the picture
              org = (center_x - 80, center_y + 60)
              font = cv2.FONT_HERSHEY_SIMPLEX
              fontScale = 0.7
              frame_rgb = cv2.putText(frame_rgb, f"angle: {int(pitch_deg)} deg", org, font, fontScale, (0, 0, 255), 2, cv2.LINE_AA)
        
        
        final_output = cv2.resize(frame_rgb, (1100, 780))  # Eventual resize  
        cv2.imshow("Video", final_output)
        cv2.waitKey(1)    

def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  node = AprilTag_Node()
  
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