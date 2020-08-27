#!/usr/bin/env python

"""
    Human Detection Service.

    ROS service that takes as input
    an image in a MAT format, and
    returns a frame with detections
    and their respective centre points.
"""

# Modules
import os
import cv2
import sys
import time
import math
import rospy
import imutils
import numpy as np
import message_filters
import face_recognition

# OpenCV-ROS bridge
from cv_bridge import CvBridge, CvBridgeError

# Messages for requests and subscriptions
from sensor_msgs.msg import Image, PointCloud2
from human_position_estimation.srv import *

#Move TIAGo to specific point
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal 
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion


from pathlib import Path
from sensor_msgs.msg import Image
from human_position_estimation.msg import *
# from human_position_estimation.srv import * # SAM commented out



class PersonDetection:

    def __init__(self):
        """
            Constructor.
        """
        # Detection target ID (person)
        self.target = 15

        # visitor detected
        self.visitor = 0
        self.confirm_attempts = 0

        # Minimum confidence for acceptable detections
        self.confidence = 0.5

        # Converted depth_image
        self.depth_image = None

        # Publishing rate
        self.rate = rospy.Rate(10)

        # Number of detections
        self.number_of_detections = 0

        # Detection messages
        self.detections = Detections()

        # Constant path
        self.path = str(Path(os.path.dirname(os.path.abspath(__file__))).parents[0])

        # Define detection's target/s
        self.targets = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        # Bounding boxes self.colours
        self.colours = np.random.uniform(0, 255, size=(len(self.targets), 3))

        # Load the neural network serialised model
        self.net = cv2.dnn.readNetFromCaffe(self.path + "/data/nn_params/MobileNetSSD_deploy.prototxt.txt",
                                            self.path + "/data/nn_params/MobileNetSSD_deploy.caffemodel")

        # Distance (human-robot distance) and detection publishers
        self.detection_pub = rospy.Publisher('detections', Detections, queue_size=1)

        print("[INFO] Successful DNN Initialisation")

        # Subscriptions (via Subscriber package)
        rgb_sub = message_filters.Subscriber("/xtion/rgb/image_rect_color", Image, queue_size=None) ## sam added queue size = None ##
        # depth_sub = message_filters.Subscriber("/xtion/depth_registered/hw_registered/image_rect_raw", Image)
        depth_sub = message_filters.Subscriber("/xtion/depth_registered/image_raw", Image, queue_size=None) ## sam added queue size = None ##


        # Synchronize subscriptions
        ats = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=1, slop=0.1) ## sam changed queue size from 5 and slop from 0.5
        ats.registerCallback(self.processSubscriptions)

        

    def detection(self, req):
        """
            Returns the frame with
            detections bounding boxes.

            Params:
                sensor_msgs/Image: Depth image syncd with RGB

            Ouput:
                int: Result of the service
        """
        # print("[INFO] Loading Image...")
        frame = self.load_img()

        # Resize image to be maximum 400px wide
        frame = imutils.resize(frame, width = 400)

        # Blob conversion (detecion purposes)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # Run feed-forward (crates detection array)
        # print("[INFO] Detection...")
        self.net.setInput(blob)
        detections = self.net.forward()

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # Get detection probability
            confidence = detections[0, 0, i, 2]

            # Get ID of the detection object
            idx = int(detections[0, 0, i, 1])

            # Filter out non-human detection with low confidence
            if confidence > self.confidence and idx == self.target:
                # Get the identity of visitor
                print('hello, visitor')
                print('Who are you?')
                print('If you are a friend: press 1')
                print('If you are postman, press 2')
                print('If you are deli-man, press 3')
                identity = input()

                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Rectangle keypoints
                top_left = (startX, startY)
                bottom_right = (endX, endY)

                # draw bounding box
                label = "{}: {:.2f}%".format(self.targets[idx], confidence * 100)
                roi = cv2.rectangle(frame, top_left, bottom_right, self.colours[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colours[idx], 2)

                # Get centre point of the rectangle and draw it
                centre_point = self.getCentre(top_left, bottom_right)
                cv2.circle(frame, centre_point, 4, (0,0,255), -1)

                # Get 480x640 ratio points
                centre_ratio_point = self.getRatioPoint(centre_point[0], centre_point[1])

                # Detection info
                detection = Detection()
                detection.ID = self.number_of_detections
                detection.centre_x = centre_ratio_point[0]
                detection.centre_y = centre_ratio_point[1]

                # Aggregate the detection to the others
                self.detections.array.append(detection)

                # Human detections counter
                self.number_of_detections += 1

                # Detect cloth colour
                red_flag = False 
                yellow_flag = False

                cloth_xs = startX
                cloth_xe = endX
                cloth_ys = (startY + endY)/2 - 80
                cloth_ye = (startY + endY)/2 - 20

                cloth_tl = (cloth_xs, cloth_ys)
                cloth_br = (cloth_xe, cloth_ye)

                cloth_box = frame[cloth_ys:cloth_ye, cloth_xs:cloth_xe]

                cv2.rectangle(frame, cloth_tl, cloth_br, (0, 255, 0), 2)

                red = np.array([(30, 30, 90),(50, 50, 130)])
                yellow = np.array([(60, 150, 150),(90, 230, 230)])

                # red = [(17, 15, 100),(50, 56, 200)]
                # yellow = [(45, 100, 135),(100, 140, 200)]
                # Recognition for friend
                if identity == 1:
                    friend_image = face_recognition.load_image_file("/tiago_ws/src/person_follower/src/person_detection/src/friend.png")
                    friend_face_encoding = face_recognition.face_encodings(friend_image)[0]
                    
                    known_face_encodings = [
                        friend_face_encoding,
                    ]
                    
                    known_face_names = [
                        "Friend"
                    ]

                    face_locations = []
                    face_encodings = []
                    face_names = []
                    process_this_frame = True

                    while True:
                        # Grab a single frame of video
                        frame = cv2.imread('/tiago_ws/src/person_follower/src/person_detection/src/face.png')

                        # Resize frame of video to 1/4 size for faster face recognition processing
                        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                        # rgb_small_frame = small_frame[:, :, ::-1]
                        rgb_small_frame = frame[:, :, ::-1]

                        # Only process every other frame of video to save time
                        if process_this_frame:
                            # Find all the faces and face encodings in the current frame of video
                            face_locations = face_recognition.face_locations(rgb_small_frame)
                            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                            face_names = []
                            for face_encoding in face_encodings:
                                # See if the face is a match for the known face(s)
                                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                                name = "Unknown"

                                # # If a match was found in known_face_encodings, just use the first one.
                                # if True in matches:
                                #     first_match_index = matches.index(True)
                                #     name = known_face_names[first_match_index]

                                # Or instead, use the known face with the smallest distance to the new face
                                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)
                                
                                if matches[best_match_index]:
                                    print('Identity confirmed. Hello friend, please come in.')
                                    self.visitor += 1
                                    name = known_face_names[best_match_index]
                                else :
                                    self.confirm_attempts += 1

                                face_names.append(name)

                        process_this_frame = not process_this_frame


                        # Display the results
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                            # top *= 4
                            # right *= 4
                            # bottom *= 4
                            # left *= 4

                            # Draw a box around the face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                            # Draw a label with a name below the face
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                        
                        # Display the resulting image
                        cv2.imshow('Video', frame)

                        # Hit 'q' on the keyboard to quit!
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                contour_area = [0,0]
                # Recognition for postman
                if identity == 2:
                    print('Hello, postman. I am verifying your identity')
                    mask_red = cv2.inRange(cloth_box, red[0], red[1])
                    output_red = cv2.bitwise_and(cloth_box, cloth_box, mask = mask_red)
                    red_gray = cv2.cvtColor(output_red,cv2.COLOR_BGR2GRAY)
                    red_ret, red_binary = cv2.threshold(red_gray,0,255,cv2.THRESH_BINARY)
                    red_binary, red_contours, red_hierarchy = cv2.findContours(red_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(output_red,red_contours,-1,(0,255,0),3)
                    if len(red_contours) > 0:
                        red_area = cv2.contourArea(red_contours[0])
                        contour_area[0] = red_area
                        red_flag = True
                        print('red', red_area)
                        print('Identity confirmed. Hello, postman. Please leave package here. Thank you.')
                        self.visitor += 1
                    else :
                        self.confirm_attempts += 1
                        print('Identity not confirmed.')
                
                # Recognition for deli-man
                elif identity == 3:
                    mask_yellow = cv2.inRange(cloth_box, yellow[0], yellow[1])
                    output_yellow = cv2.bitwise_and(cloth_box, cloth_box, mask = mask_yellow)
                    yellow_gray = cv2.cvtColor(output_yellow,cv2.COLOR_BGR2GRAY)
                    yellow_ret, yellow_binary = cv2.threshold(yellow_gray,0,255,cv2.THRESH_BINARY)
                    yellow_binary, yellow_contours, yellow_hierarchy = cv2.findContours(yellow_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(output_yellow,yellow_contours,-1,(0,255,0),3)
                    if len(yellow_contours) > 0:
                        yellow_area = cv2.contourArea(yellow_contours[0])
                        contour_area[1] = yellow_area
                        yellow_flag = True
                        print('yellow', yellow_area)
                        print('Identity confirmed. Hello, deli-man. Please follow me')
                        try:
                            
                            navigator = GoToPose()
                            x = 0
                            y = 3
                            theta = 0
                            position = {'x': x, 'y' : y}
                            quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : np.sin(theta/2.0), 'r4' : np.cos(theta/2.0)}
                            rospy.loginfo("Go to (%s, %s) pose", position['x'], position['y'])
                            success = navigator.goto(position, quaternion)

                            if success:
                                rospy.loginfo("Please leave package here.")
                                
                                while True:
                                    visitor_cmd = input("When you finish, press 1 to send me back to the door. Thank you.")
                                    if visitor_cmd == 1:

                                        try:
                                            navigator = GoToPose()
                                            x = 0
                                            y = 0
                                            theta = 0
                                            position = {'x': x, 'y' : y}
                                            quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : np.sin(theta/2.0), 'r4' : np.cos(theta/2.0)}
                                            rospy.loginfo("Go to (%s, %s) pose", position['x'], position['y'])
                                            success = navigator.goto(position, quaternion)
                                            self.visitor += 1
                                            break

                                        except rospy.ROSInterruptException:
                                            rospy.loginfo("Quitting")
                                    
                                    elif visitor_cmd != 1:
                                        print('Please press 1')
                    
                        except rospy.ROSInterruptException:
                            rospy.loginfo("Quitting")
                            rospy.sleep(1)

                    else:
                        self.confirm_attempts += 1
                        print('Identity not confirmed')

                #Press wrong button
                else:
                    print('Please press correct button') 

                if self.confirm_attempts >= 5:
                    print('Seems you are a stranger for me. Please leave or contact the host')

                if self.visitor >= 3:
                    print('all visitor confirmed!!')
                

                # if len(cloth_box):
                #     cv2.imshow('frame',frame)
                    
                #     cv2.imshow('cloth', np.hstack([cloth_box,output_red,output_yellow]))

                
        # Save frame
        # cv2.imshow('frame',frame)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        self.store(frame)
        

        # Add number_of_detections item to the detections message
        self.detections.number_of_detections = self.number_of_detections

        # Request depth mapping for the detections
        # rospy.loginfo("Requesting depth mapping for the detections...")
        # self.requestMapping(self.detections, req.depth)
        #
        # return RequestDetectionResponse("success")
        self.requestMapping(self.detections)
        # return("Success")


        

    def requestMapping(self, detections):
        """
            ROS service that requests the
            depth distance of the detections
            from the RGBD-optical frame.

            Arguments:
                detections: RGB detections rgb position
        """
        # Wait for service to come alive
        # rospy.wait_for_service('detection_pose')

        try:
            # Build request
            # request = rospy.ServiceProxy('detection_pose', RequestDepth)
            #
            # # Get response from service
            # response = request(detections, depth_image)

            # print("Publishing detections")

            # Publish detections
            self.detections.header.stamp = rospy.Time.now() ## ADDED BY SAM ##
            self.detection_pub.publish(self.detections)

            # Access the response field of the custom msg
            # rospy.loginfo("Pose service: %s", response.res)

            # Clean
            self.detections = Detections()
            self.number_of_detections = 0

        except rospy.ServiceException as e:
            rospy.loginfo("Depth service call failed: %s", e)

    def getDetectionObject(self, confidence, rgb_x, rgb_y):
        """
            Detection object for
            detection custom message
            population.

            Arguments:
                param1: detection confidence
                param2: x coordinate of the centre box
                param3: y coordinate of the centre box

            Returns:
                object: detection object
        """
        return {'confidence': confidence, 'rgb_x': rgb_x, 'rgb_y': rgb_y}

    # Load image to be processed
    def load_img(self):
        """
            Load image to be processed.

            Returns:
                image: MAT image
        """
        return cv2.imread(self.path + "/data/converted/image.png")

    def store(self, frame):
        """
            Stores image with
            detections' bounding
            boxes.

            Arguments:
                param1: MAT image with detection boxes
        """
        cv2.imwrite(self.path + "/data/detections/detections.png", frame)

    def getCentre(self, tl, br):
        """
            Finds centre point of the
            bounding box with respect
            to the 300x300 ratio.

            Arguments:
                int: Top left corner of the rectangle
                int: Bottom right corner of the rectangle

            Returns:
                tuple of ints: X and Y coordinate of centre point
        """
        # Compute distances
        width  = br[0] - tl[0]
        height = br[1] - tl[1]

        # Return centre
        return (tl[0] + int(width * 0.5), tl[1] + int(height * 0.5))

    def getRatioPoint(self, x, y):
        """
            Find point in the
            480x640 ratio.

            Arguments:
                int: 400x300 X coordinate (width)
                int: 400x300 Y coordinate (height)

            Returns:
                tuple of ints: X and Y coordinate of centre point
        """
        # return (int((x/400) * 640), int((y/300) * 480))
        ## sam ##
        return (int((x*640) / 400), int((y*480) / 300))








    # Constant path
    PATH = str(Path(os.path.dirname(os.path.abspath(__file__))).parents[0])

    # def requestDetection(req):
    #     """
    #         Sends a service request to
    #         the person detection module.
    #
    #         Arguments:
    #             sensor_msgs/Image: Depth image
    #
    #         Returns:
    #             string: Service response
    #     """
    #     # Wait for service to come alive
    #     rospy.wait_for_service('detection')
    #
    #     try:
    #         # Build request
    #         request = rospy.ServiceProxy('detection', RequestDetection)
    #
    #         # Get response from service
    #         response = request(req)
    #
    #         # Access the response field of the custom msg
    #         rospy.loginfo("Detection service: %s", response.res)
    #         # return response.res
    #
    #     except rospy.ServiceException as e:
    #         rospy.loginfo("Detection service call failed: %s", e)

    def store(self, cv_image):
        """
            Stores the converted raw image from
            the subscription and writes it to
            memory.

            Arguments:
                MAT: OpenCV formatted image
        """
        cv2.imwrite(self.PATH + "/data/converted/image.png", cv_image)

    def toMAT(self, rgb_image):
        """
            Converts raw RGB image
            into OpenCV's MAT format.

            Arguments:
                sensor_msgs/Image: RGB raw image

            Returns:
                MAT: OpenCV BGR8 MAT format
        """
        try:
            cv_image = CvBridge().imgmsg_to_cv2(rgb_image, 'bgr8')
            return cv_image

        except Exception as CvBridgeError:
            print('Error during image conversion: ', CvBridgeError)

    def processSubscriptions(self, rgb_image, depth_image):
        """
            Callback for the TimeSynchronizer
            that receives both the rgb raw image
            and the depth image, respectively
            running the detection module on the
            former and the mappin process on the
            former at a later stage in the chain.

            Arguments:
                sensor_msgs/Image: The RGB raw image
                sensor_msgs/Image: The depth image
        """
        # print("Got depth and rgb.")
        # Processing the rgb image
        rgb_cv_image = self.toMAT(rgb_image)
        self.store(rgb_cv_image)

        # Request services
        # rospy.loginfo("Requesting detection and mapping services")
        # requestDetection(depth_image)
        self.detection(depth_image)

        # print(rgb_cv_image)
        # frame_sub = rospy.Subscriber('/xtion/rgb/image_raw', Image, queue_size=None)
        # frame_sub2 = CvBridge().imgmsg_to_cv2(frame_sub,"bgr8")


        # This part is for face recognition


class GoToPose():
    def __init__(self):

        self.goal_sent = False

        # What to do if shut down (e.g. Ctrl-C or failure)
        rospy.on_shutdown(self.shutdown)
        
        # Tell the action client that we want to spin a thread by default
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Wait for the action server to come up")

        # Allow up to 5 seconds for the action server to come up
        self.move_base.wait_for_server(rospy.Duration(5))

    def goto(self, pos, quat):

        # Send a goal
        self.goal_sent = True
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        # goal.target_pose.pose = Pose(Point(pos['x'], pos['y'], 0.000),
        #                                 Quaternion(quat['r1'], quat['r2'], quat['r3'], quat['r4']))
        goal.target_pose.pose.position.x = pos['x']
        goal.target_pose.pose.position.y = pos['y']
        goal.target_pose.pose.orientation = Quaternion(quat['r1'], quat['r2'], quat['r3'], quat['r4'])

        # Start moving
        self.move_base.send_goal(goal)

        # Allow TurtleBot up to 60 seconds to complete task
        success = self.move_base.wait_for_result(rospy.Duration(60)) 

        state = self.move_base.get_state()
        result = False

        if success and state == GoalStatus.SUCCEEDED:
            # We made it!
            result = True
        else:
            self.move_base.cancel_goal()

        self.goal_sent = False
        return result

    def shutdown(self):
        if self.goal_sent:
            self.move_base.cancel_goal()
        rospy.loginfo("Stop")
        rospy.sleep(1)




    


def main(args):

    # Initialise node
    rospy.init_node('person_detection', anonymous=True)
    # rospy.init_node('nav_test',anonymous=True)

    try:
        # Initialise
        hd = PersonDetection()

        # Detection service
        # service = rospy.Service('detection', RequestDetection, hd.detection)

        # Spin it baby !
        rospy.spin()

    except KeyboardInterrupt as e:
        print('Error during main execution' + e)

    cv2.destroyAllWindows()


# Execute main
if __name__ == '__main__':
    main(sys.argv)