# Thesis work
This repository contains the ROS2 nodes developed for my thesis work, a robot for indoor service robotics. 
The first part involved the detection of a person in the camera streaming via a neural network and the derivation of the relative distance with respect to the robot. Files sort.py, posenet_engine_mod.py and posenet_node.py are related to this part. To run them Coral USB accelerator is required[^1]

Then visual relocalization was performed using April tags[^2]. Knowing a priori the pose of the marker in a global map, the absolute pose of the robot can be estimated computing the relative distance to the tag when it is framed. In this way the odometry error accumulated while roaming is corrected. The april_tag.py script is the only used in this part.

The third ROS node makes the robot spin and acquire a database of 10 images. Then on the user will, it spins acquiring another set of 10 images. Features from the environment are extracted with SIFT and RANSAC is applied to filter inlier couples. This way the essential matrix is estimated and the different camera angle is computed. Combined with depth mesures of features, the robot different pose is estimated. Files features.py and feat_extract.py are related to this part

Every application involves the use of ROS2 Foxy. The camera used was Intel RealSense D435i, and its ros package is required [^3].

[^1]: https://coral.ai/docs/accelerator/get-started/#requirements
[^2]: https://april.eecs.umich.edu/software/apriltag
[^3]: https://github.com/IntelRealSense/realsense-ros/tree/foxy
