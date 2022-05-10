import sys
import argparse
import math

from charset_normalizer import detect

from .Camera import Camera

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

from rmf_obstacle_msgs.msg import Obstacles, Obstacle, ObstacleData

class HumanDetector(Node):
  def __init__(self,visualize=False):
      super().__init__('human_detector_node')
      self.get_logger().info("Starting human detection...")
      self.camera = Camera(visualize)
      self.camera.start()
      if (self.declare_parameter('frame_id', 'camera_base_link').value):
          self.frame_id = self.get_parameter('frame_id').value
      self.level_name = ""
      if (self.declare_parameter('level_name', 'L1').value):
          self.level_name = self.get_parameter('level_name').value
      if (self.declare_parameter('camera_pose', [0,0,0,0,0,0]).value):
          self.camera_pose = self.get_parameter('camera_pose').value

      self.timer = self.create_timer(
        1.0,
        self._timer_cb)

      self.pub = self.create_publisher(
        Obstacles,
        'rmf_obstacles',
        qos_profile=qos_profile_system_default)

      # todo(YV): Publish static tf

  def _timer_cb(self):
      msg = Obstacles()
      id = 0
      for detection in self.camera.detections:
          _msg = Obstacle()
          _msg.header.frame_id = self.frame_id
          _msg.header.stamp = self.get_clock().now().to_msg()
          # todo
          _msg.id = id
          id = id = 1
          _msg.source = "human_detector"
          _msg.level_name = self.level_name
          _msg.classification = "human"
          # todo(YV): Apply transformation based on camera_pose
          # ObstacleData
          _msg.data.box.center.position.x = detection.depth_x
          _msg.data.box.center.position.y = detection.depth_z
          _msg.data.box.center.position.z = detection.depth_y
          _msg.data.box.size.x = detection.width
          _msg.data.box.size.y = detection.width
          _msg.data.box.size.z = detection.height

          _msg.lifetime.sec = 1
          _msg.lifetime.nanosec  = 0
          _msg.action = _msg.ACTION_ADD
          msg.obstacles.append(_msg)
      if len(msg.obstacles) > 0:
          self.pub.publish(msg)

def main(argv=sys.argv):
    rclpy.init(args=argv)
    args_without_ros = rclpy.utilities.remove_ros_args(argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', help='Debug visualization')
    args = parser.parse_args(args_without_ros[1:])

    n = HumanDetector(args.debug)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass