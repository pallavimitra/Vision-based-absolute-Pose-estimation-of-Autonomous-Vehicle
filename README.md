# Vision based absolute Pose estimation of Autonomous Vehicle

Accurate localization for any autonomous vehicle (AV), especially in indoor and GPS-denied (Global Positioning System) environment. To close the gap, vision-based pose estimation in real time is a promising solution. The aim of this work is to provide a solution to estimate the pose of vehicles (6 degrees of freedom) from a single image in a real-world traffic environment. This 6 DoF includes 3D world coordinates (x,y,z), yaw, pitch, and roll of the detected car as follows:
<p align="center">
  <img src="images/motion_yaw_pitch_roll.jpg" />
</p>
A two-staged 6DoF object detection pipeline is proposed in this work. Firstly, a resnet like architicture (YOLO is applied to provide object bounding boxes. Then two regressor is applied to estimate the 3d properties and euler angles.
Below is the architecture description for the same.
