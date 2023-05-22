import numpy as np
import argparse
import cv2
import os
from helpers import convert_and_trim_bb
import dlib
import time

# With OpenCV and Deep Neural Network (DNNs)
def face_detect_dnn(args):
    print("[INFO] Loading dnn face detector")
    prototxt_file = "deploy.prototxt.txt"
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    deploy_prototxt_path = os.path.join('../model/', prototxt_file)
    caffe_model_path = os.path.join('../model/', model_file)
    net = cv2.dnn.readNetFromCaffe(deploy_prototxt_path, caffe_model_path)

    # load input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    image_path = os.path.join('../data/', args["image"])
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    print("[INFO] computing face detection..")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
    
            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Face Detect", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# With OpenCV and Haar cascades
def face_detect_haar_cascades(args):
    print("[INFO] loading haar cascades face detector...")
    cascade_file = "haarcascade_frontalface_default.xml"
    cascade_path = os.path.join('../model/', cascade_file)
    detector = cv2.CascadeClassifier(cascade_path)

    # load the input image from disk, resize it, and convert it to grayscale
    image_path = os.path.join('../data/', args["image"])
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the input image using the haar cascade face detector
    print("[INFO] performing face detection...")
    rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    print("[INFO] {} faces detected...".format(len(rects)))

    # loop over the bounding boxes
    for (x, y, w, h) in rects:
        # draw the face bounding box on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# With dlib and the HOG + Linear SVM algorithm
def face_detect_dlib_hog(args):
    # load dlib's HOG + Linear SVM face detector
    print("[INFO] loading HOG + Linear SVM face detector...")
    detector = dlib.get_frontal_face_detector()
    
    # load the input image from disk, resize it, and convert it from
    # BGR to RGB channel ordering (which is what dlib expects)
    image_path = os.path.join('../data/', args["image"])
    image = cv2.imread(image_path)
    image = cv2.resize(image, (600, 600))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # perform face detection using dlib's face detector
    start = time.time()
    print("[INFO[ performing face detection with dlib...")
    rects = detector(rgb, args["upsample"])
    end = time.time()
    print("[INFO] face detection took {:.4f} seconds".format(end - start))

    # convert the resulting dlib rectangle objects to bounding boxes,
    # then ensure the bounding boxes are all within the bounds of the
    # input image
    boxes = [convert_and_trim_bb(image, r) for r in rects]
    # loop over the bounding boxes
    for (x, y, w, h) in boxes:
        # draw the bounding box on our image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

# With dlib and the max-margin object detector (MMOD)
def face_detect_dlib_mmod(args):
    # load dlib's CNN face detector
    print("[INFO] loading CNN face detector...")
    mmod_human_face_detector_file = "mmod_human_face_detector.dat"
    mmod_human_face_detector_path = os.path.join('../model/', mmod_human_face_detector_file)
    detector = dlib.cnn_face_detection_model_v1(mmod_human_face_detector_path)

    # load the input image from disk, resize it, and convert it from
    # BGR to RGB channel ordering (which is what dlib expects)
    image_path = os.path.join('../data/', args["image"])
    image = cv2.imread(image_path)
    image = cv2.resize(image, (600, 600))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # perform face detection using dlib's face detector
    start = time.time()
    print("[INFO[ performing face detection with dlib...")
    results = detector(rgb, args["upsample"])
    end = time.time()
    print("[INFO] face detection took {:.4f} seconds".format(end - start))

    # convert the resulting dlib rectangle objects to bounding boxes,
    # then ensure the bounding boxes are all within the bounds of the
    # input image
    boxes = [convert_and_trim_bb(image, r.rect) for r in results]
    
    # loop over the bounding boxes
    for (x, y, w, h) in boxes:
        # draw the bounding box on our image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-m", "--method", required=True, help="Choose method to determine face detection: 1. Dnn, 2. Haar Cascades, 3. HOG + Linear SVM, 4. Dlib's CNN ")
    # ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' protoxt file")
    # ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-u", "--upsample", type=int, default=1, help="# of times to upsample")
    args = vars(ap.parse_args())
    print("[INFO] Loading model...")
    if(args["method"].lower() == "dnn"):
        face_detect_dnn(args)
    elif(args["method"].lower() == "haarcascades"):
        face_detect_haar_cascades(args)
    elif(args["method"].lower() == "hog"):
        face_detect_dlib_hog(args)
    elif(args["method"].lower() == "mmod"):
        face_detect_dlib_mmod(args)

if __name__ == '__main__':
    main()

