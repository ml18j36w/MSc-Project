# MSc-Project
# About this project
This project is for welcoming visitor tasks. The robot will recognise different visitors and act according to the identity of visitors.

# Pre-request

1. ROS
2. Python (recommand Python2)
3. face_recognition library

# Package functions

person_follower contains the main function of packages for welcoming visitors. 

pal_face_detector_opencv is used for detecting human faces and storing them for recognition. The purpose of this package is to capture human face more accuratly.

# How to use

There are pre-stored human faces in the person_follower repository for testing. If you want test different human face, simply change the value of "friend_image" and "frame" in welcoming.py into the images of known face and the face you want to be recognised. 

1. use pal_face_detector_opencv to capture faces from robot camera
2. use person_follower to recognise visitors
