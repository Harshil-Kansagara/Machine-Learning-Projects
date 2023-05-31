import cv2
import os
import time

def object_tracking():
    # create a tracker
    tracker = cv2.TrackerGOTURN_create()

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Failed to open video capture")
        return

    bounding_box = None

    while(True):
        ret, frame = capture.read()

        # Break the loop if no more frames are available
        if not ret:
            break

        if bounding_box is not None:
            (success, box) = tracker.update(frame)

            if success:
                # Draw the bounding box on the frame
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # if the 's' key is selected, we are going to "select" a bounding box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make sure you press ENTER or SPACE after selecting the ROI)
            bounding_box = cv2.selectROI("Frame", frame, False)
            # start OpenCV object tracker using the supplied bounding box coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, bounding_box)

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

object_tracking()