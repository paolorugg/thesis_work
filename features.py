import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np
import math
import time
from geometry_msgs.msg import Twist
from tesi.feat_extract import *



class FeatMatching_Node(Node):   
        
    def __init__(self):

        super().__init__('node')
        
        self.subscription1 = self.create_subscription(Image, '/camera/color/image_raw', self.acquire_color_image, 10)
        self.subscription2 = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.acquire_depth_image, 10)
        self.subscription1
        self.subscription2
        
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.br = CvBridge()
        self.sift = cv2.SIFT_create()        
        
        self.twist_rot = Twist()
        self.twist_rot.angular.x = 0.0
        self.twist_rot.angular.y = 0.0
        self.twist_rot.angular.z = math.pi/19.6 # Rotate of 36° in 3.95 seconds, and considering acceleration
        
        self.twist_stop = Twist()
        self.twist_stop.angular.x = 0.0
        self.twist_stop.angular.y = 0.0
        self.twist_stop.angular.z = 0.0 # Command to stop the robot
        
        self.gray_image_list=[]
        self.depth_image_list=[]
        self.kp_list=[]
        self.des_list=[]
        self.step = 0  
        self.flag1 = False
        self.flag2 = False
        print('fine init')

    
    def timer_callback(self):

        if self.step < 10 and self.flag1 and self.flag2:   # Initialization fo the database of images
            tmp_bgr_image = self.br.imgmsg_to_cv2(self.color_image)
            gray_image= cv2.cvtColor(tmp_bgr_image,cv2.COLOR_BGR2GRAY)              
            self.gray_image_list.append(gray_image)
            self.flag1 = False
            cv2.imwrite('image_queue_'+str(self.step)+'.png', gray_image) 
            
            depth_image=self.br.imgmsg_to_cv2(self.depth_image_raw, '32FC1')
            self.depth_image_list.append(depth_image)
            self.flag2 = False
            np.savetxt('immagini_depth'+str(self.step)+'.csv', depth_image, delimiter=',')
          
            kp, des = self.sift.detectAndCompute(gray_image, None)
            self.kp_list.append(kp)
            self.des_list.append(des)
            time.sleep(0.5)
            
            self.publisher_.publish(self.twist_rot)
            
            time.sleep(3.99)
            
            self.publisher_.publish(self.twist_stop)
            self.step += 1
            print('Robot has rotated of ' + str(self.step*36) + ' degrees')
            
            time.sleep(0.5)
        if self.step == 10 and self.flag1 and self.flag2: 
          print('Descriptors computed. Waiting to return to zero position')
          input('press ENTER to acquire query images')
          self.flag1 = False
          self.flag2 = False
          self.step += 1
        if self.step > 10 and self.step <= 20 and self.flag1 and self.flag2: 
          tmp_bgr_image = self.br.imgmsg_to_cv2(self.color_image)
          gray_image = cv2.cvtColor(tmp_bgr_image,cv2.COLOR_BGR2GRAY)    
          self.gray_image_list.append(gray_image)
          self.flag1 = False
          cv2.imwrite('image_queue_'+str(self.step-1)+'.png', gray_image)               

          depth_image =self.br.imgmsg_to_cv2(self.depth_image_raw, '32FC1')
          self.depth_image_list.append(depth_image)
          self.flag2 = False
          np.savetxt('immagini_depth'+str(self.step-1)+'.csv', depth_image, delimiter=',')
          
          kp, des = self.sift.detectAndCompute(gray_image, None)
          self.kp_list.append(kp)
          self.des_list.append(des)
          time.sleep(0.5)
          
          self.publisher_.publish(self.twist_rot)
          
          time.sleep(3.99)
          
          self.publisher_.publish(self.twist_stop)
          self.step += 1
          print('Robot has rotated of ' + str((self.step-1)*36-360) + ' degrees')
          
          time.sleep(0.5)
       
        if  self.step == 21 and self.flag1 and self.flag2: 
          max_features_matched_best = 0
          max_features_matched_second_best = 0
          max_features_matched_third_best = 0
          db_query_index_best = [50,50] #just initialize
          db_query_index_second_best = [50,50]
          db_query_index_third_best = [50,50]
          src_pts_best=[]
          dst_pts_best=[]
          src_pts_second_best=[]
          dst_pts_second_best=[]
          src_pts_third_best=[]
          dst_pts_third_best=[]
          for i in range(10,20):
            des_query = self.des_list[i]
            kp_query = self.kp_list[i]
            for j in range(0,10):
              des_db = self.des_list[j]
              kp_db = self.kp_list[j]
              if des_query is not None and kp_query is not None and des_db is not None and kp_db is not None:
                src_temp, dst_temp, feat_matched = siftMatching(kp_db, des_db, kp_query, des_query)
                if feat_matched > max_features_matched_third_best:
                  if feat_matched > max_features_matched_second_best:
                    if feat_matched > max_features_matched_best:
                      src_pts_third_best = src_pts_second_best # There is a new best. Others go down one position
                      dst_pts_third_best = dst_pts_second_best
                      db_query_index_third_best = db_query_index_second_best
                      max_features_matched_third_best = max_features_matched_second_best
                      
                      src_pts_second_best = src_pts_best
                      dst_pts_second_best = dst_pts_best
                      db_query_index_second_best = db_query_index_best
                      max_features_matched_second_best = max_features_matched_best
                      
                      src_pts_best = src_temp
                      dst_pts_best = dst_temp
                      db_query_index_best = [j,i]
                      max_features_matched_best = feat_matched
                    else : #There is a new second best
                      src_pts_third_best = src_pts_second_best 
                      dst_pts_third_best = dst_pts_second_best
                      db_query_index_third_best = db_query_index_second_best
                      max_features_matched_third_best = max_features_matched_second_best
                      
                      src_pts_second_best = src_temp
                      dst_pts_second_best = dst_temp
                      db_query_index_second_best = [j,i]
                      max_features_matched_second_best = feat_matched
                  else: #There is a new third best
                    src_pts_third_best = src_temp 
                    dst_pts_third_best = dst_temp
                    db_query_index_third_best = [j,i]
                    max_features_matched_third_best = feat_matched      
              j += 1
            i += 1
          
          self.step += 1
          pixels_db_best, pixels_query_best, theta_best, z1_best, x1_best, z2_best, x2_best = self.get_x_y_angle(src_pts_best, dst_pts_best, db_query_index_best[0], db_query_index_best[1])
          print('\n')
          pixels_db_second_best, pixels_query_second_best, theta_second_best, z1_second_best, x1_second_best, z2_second_best, x2_second_best = self.get_x_y_angle(src_pts_second_best, dst_pts_second_best, db_query_index_second_best[0], db_query_index_second_best[1])
          print('\n')
          pixels_db_third_best, pixels_query_third_best, theta_third_best, z1_third_best, x1_third_best, z2_third_best, x2_third_best = self.get_x_y_angle(src_pts_third_best, dst_pts_third_best, db_query_index_third_best[0], db_query_index_third_best[1])

          # Computing the difference between feature position in pixels in the two frame
          best = abs(pixels_db_best[0] - pixels_query_best[0]) + abs(pixels_db_best[1] - pixels_query_best[1])
          second_best = abs(pixels_db_second_best[0] - pixels_query_second_best[0]) + abs(pixels_db_second_best[1] - pixels_query_second_best[1])
          third_best = abs(pixels_db_third_best[0] - pixels_query_third_best[0]) + abs(pixels_db_third_best[1] - pixels_query_third_best[1])
          # The image with the smallest difference is taken
          if best < second_best and best < third_best and z1_best < 4 and z2_best < 4:
            x_algo, y_algo, theta_algo = self.compute_absolute_position(theta_best, z1_best, x1_best, z2_best, x2_best, db_query_index_best[0], db_query_index_best[1])
          elif second_best < best and second_best < third_best and z1_second_best < 4 and z2_second_best < 4:
            x_algo, y_algo, theta_algo = self.compute_absolute_position(theta_second_best, z1_second_best, x1_second_best, z2_second_best, x2_second_best, db_query_index_second_best[0], db_query_index_second_best[1])
          elif z1_third_best < 4 and z2_third_best < 4:
            x_algo, y_algo, theta_algo = self.compute_absolute_position(theta_third_best, z1_third_best, x1_third_best, z2_third_best, x2_third_best, db_query_index_third_best[0], db_query_index_third_best[1])
          else:
            print('No useful features found')
            return 0
          
          print('In the end it is found:')
          print('x_final =', x_algo)
          print('y_final =', y_algo)
          print('theta_final =', theta_algo)
      
          
        
        

    def acquire_color_image(self, data):
        self.color_image = data
        self.flag1 = True
           
    def acquire_depth_image(self, data):
        self.depth_image_raw = data
        self.flag2 = True
        
    def get_x_y_angle(self, src_pts, dst_pts, db_index, query_index):
        print('db_index =', db_index)
        print('query_index =', query_index)
        depth_image_db = self.depth_image_list[db_index]
        depth_image_query = self.depth_image_list[query_index]
        angle, src, dst = recover_angle(src_pts, dst_pts)
        print('angle query->db = ', angle)
        
        # Get z1,x1,z2,x2 relative to the nearest feature
        min_distance_db = 10000
        feat_index = 1000 # Random numbers just to initialize
        for j in range(src.shape[0]):
          distance = depth_image_db[round(src[j][1]), round(src[j][0])]
          if distance < min_distance_db and distance > 0:
            if depth_image_query[round(dst[j][1]), round(dst[j][0])] > 0:
              feat_index = j
              min_distance_db = distance
          j += 1
        if feat_index == 1000:
          return 10000, 10000, 0, 10000, 10000 # It can rarely happen that all features are too far or too close for the depth camera, so their distance is of no use. In this case discard the couple of images
        else :
          pixels_db = src[feat_index]
          pixels_query = dst[feat_index]
          print('feat_pixels_ db =', pixels_db)
          print('feat_pixels_query =', pixels_query)
          # Distances in meters
          z1 = min_distance_db/1000 
          x1 = (src[feat_index][0]-316.72235107421875) * z1 / 617.4224243164062
          z2 = depth_image_query[round(dst[feat_index][1]), round(dst[feat_index][0])]/1000 
          x2 = (dst[feat_index][0]-316.72235107421875) * z2 / 617.4224243164062
          print('feat distance_db =', z1)
          print('feat distance_query =', z2)
          print('x1 =',x1)
          print('x2 =',x2)
          return pixels_db, pixels_query, angle, z1, x1, z2, x2
        
    def compute_absolute_position(self, angle, z1, x1, z2, x2, db_index, query_index):     
        # Find deltaz and deltax (in the reference system of the db image)
        angle_rad = angle * math.pi / 180
        deltaz = z1 + 0.1 - (x2* math.sin(angle_rad) + (z2 + 0.1)*math.cos(angle_rad))
        deltax = x1 - (x2 * math.cos(angle_rad) - (z2 + 0.1)*math.sin(angle_rad))
        
        #Final refinement. Bring back the rotated reference system to one parallel to global and centered in (0,0)
        deltax_rot = deltaz
        deltay_rot = -deltax
        print('rotat_x =', deltax_rot)
        print('rotat_y =', deltay_rot)
        
        theta =  36 * db_index
        theta_rad = theta * math.pi / 180
        #R = [[math.cos(theta), -math.sin(theta)],
        #     [math.sin(theta), math.cos(theta)]]
        final_displacement_x = math.cos(theta_rad) * deltax_rot - math.sin(theta_rad) * deltay_rot
        final_displacement_y = math.sin(theta_rad) * deltax_rot + math.cos(theta_rad) * deltay_rot
        final_angle = (theta + angle - 36 * (query_index-10))%360 # e.g. second query with index 11 in the list is shifted of 36° from the position of arrival in (0,0), so 36 degrees should be
        return final_displacement_x, final_displacement_y, final_angle
    

def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  node = FeatMatching_Node()

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