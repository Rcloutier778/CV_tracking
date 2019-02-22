# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import math
import cv2
from multiprocessing import Pool,Queue

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    ap.add_argument("--fps",type=int, default=15,
        help="frame rate")
    ap.add_argument("-a", "--min-area", type=int, default=500,
                    help="minimum area size")
    ap.add_argument("-i","--motion-tracking",type=bool,default=True,
                    help="Enable motion tracking")
    ap.add_argument("-o", "--object-tracking", type=bool, default=True,
                    help="Enable selectable object tracking")
    ap.add_argument("-t","--tracker",type=str,default="kcf",
                    help="Tracker type for object tracking")
    ap.add_argument("-f", "--facial_recog", type=bool, default=True,
                    help="Facial Recognition Tracking")
    args = vars(ap.parse_args())
    
    
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    if args['facial_recog']:
        (H, W) = (None, None)
        # initialize our centroid tracker and frame dimensions
        ct = CentroidTracker()

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        # initialize the video stream and allow the camera sensor to warmup
        
    if args['motion_tracking']:
        firstFrame=None
        
    if args["object_tracking"]:
        initBB = None
        # extract the OpenCV version info
        (major, minor) = cv2.__version__.split(".")[:2]
    
        # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
        # function to create our object tracker
        if int(major) == 3 and int(minor) < 3:
            tracker = cv2.Tracker_create(args["tracker"].upper())
    
        # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
        # approrpiate object tracker constructor:
        else:
            OPENCV_OBJECT_TRACKERS = {
                "csrt": cv2.TrackerCSRT_create,
                "kcf": cv2.TrackerKCF_create,
                "boosting": cv2.TrackerBoosting_create,
                "mil": cv2.TrackerMIL_create,
                "tld": cv2.TrackerTLD_create,
                "medianflow": cv2.TrackerMedianFlow_create,
                "mosse": cv2.TrackerMOSSE_create
            }
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

    while True:
        st=time.time()
        #read the next frame from the video stream and resize it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        
        if args['motion_tracking']:
            
            firstFrame=motionTracker(frame.copy(),firstFrame,args["min_area"])
            
        if args['object_tracking']:
            initBB=objectTracker(frame.copy(),initBB,tracker)
        
        if args['facial_recog']:
            W,H = facialRecogTracker(frame.copy(),net,ct,W,H,args["confidence"])

        sleepTime = (1 / args['fps'])-(time.time()-st)
        if sleepTime > 0:
            time.sleep(sleepTime)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    
    
def facialRecogTracker(frame,net,ct,W,H,confidence):
    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))
            # draw a bounding box surrounding the object so we can
            # visualize it
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
    
    # update our centroid tracker using the computed set of bounding
    # box rectangles
    try:
        objects = ct.update(rects)
    except RuntimeError:
        return W,H
    
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    
    # show the output frame
    cv2.imshow("Facial Recognition", frame)
    return W,H



def objectTracker(otFrame,initBB,tracker):
    #otFrame = frame.copy()
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(otFrame)
    
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(otFrame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
    
        info = [
            ("Tracker", tracker),
            ("Success", "Yes" if success else "No"),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(otFrame, text, (10, otFrame.shape[:2][0] - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # show the output frame
    cv2.imshow("Object Tracker Frame", otFrame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to 1`track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Object Tracker Frame", otFrame, fromCenter=False,
                               showCrosshair=True)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(otFrame, initBB)
    return initBB

def motionTracker(motionFrame,firstFrame,min_area):
    gray = cv2.cvtColor(motionFrame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # if the frame dimensions are None, grab them
    
    if firstFrame is None:
        firstFrame = gray
        return firstFrame
    tempFirstFrame = gray.copy()
    # compute the absolute difference between the current frame and
    # first frame
    
    frameDelta = cv2.absdiff(firstFrame, gray)
    
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    trackedChanges = []
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, 0.0, True)
        cv2.fillPoly(motionFrame, [approx], 255)
        trackedChanges.append(c)
    # draw timestamp on the frame
    # show the frame and record if the user presses a key
    
    firstFrame = tempFirstFrame.copy()
    cv2.imshow("Motion Poly", motionFrame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    return firstFrame


if __name__ =='__main__':
    main()
