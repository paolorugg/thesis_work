from custom_interfaces.srv import CombinePoseGoal
from custom_interfaces.msg import PoseGoal
from nav_msgs.msg import Odometry 
from std_msgs.msg import Float32MultiArray

import rclpy
from rclpy.node import Node


class ServicePoseGoal(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(CombinePoseGoal, 'combine_pose_and_goal', self.combine_pose_goal_callback)
        self.subscription1 = self.create_subscription(Odometry, '/odom', self.odom_listener_callback, 10)
        self.subscription1
        self.subscription2 = self.create_subscription(Float32MultiArray, '/distance_with_depth_camera', self.goal_listener_callback, 10)
        self.subscription2
        

    def combine_pose_goal_callback(self, request, response):
        response.output.pose = self.pose_from_odom
        response.output.goal = self.goal_from_topic
        print('response is= ',response)
        self.get_logger().info('Output of the service: %r' %(response,))
        return response

    def odom_listener_callback(self, data):
        self.pose_from_odom = data.pose.pose
        
    def goal_listener_callback(self, data):
        self.goal_from_topic = data
    

def main(args=None):
    rclpy.init(args=args)

    service = ServicePoseGoal()

    rclpy.spin(service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()