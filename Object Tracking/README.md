# Object Tracking #
Object tracking is a computer vision application where a program detects objects  and then tracks their movements in space or across different camera angles. Object tracking can identify and follow multiple objects in an image. For example, a football recording studio could follow where a ball is in a photo.

Object tracking a significant computer vision technology popular in augmented reality for estimating or predicting the positions and other applicable information of moving objects in real time.

### Compared and Contrasted: Object detection vs. object tracking ###
Object detection algorithms identify objects in an image or video and their location in the media. This can be an algorithm on its own, or used to enable object tracking. Object tracking algorithms, on the other hand, follow objects over frames in a video.

### How object tracking works ###
Here is a high-level overview of how object tracking works:
1. **Object Detection:** Object tracking typically starts with an initial object detection step. In this step, a bounding box or region of interest (ROI) is drawn around the target object in the first frame of the video or image sequence. Various object detection algorithms can be used for this purpose, such as Haar cascades, HOG (Histogram of Oriented Gradients), or deep learning-based detectors like YOLO (You Only Look Once) or Faster R-CNN.

2. **Feature Extraction:** Once the initial detection is performed, features are extracted from the object within the bounding box. These features can include color histograms, texture descriptors, or visual keypoints. The choice of features depends on the specific tracking algorithm being used.

3. **Motion Estimation:** The next step is to estimate the motion of the object between consecutive frames. This can be done by comparing the extracted features or by calculating optical flow, which measures the displacement of pixels between frames. Motion estimation helps predict the position of the object in the next frame.

4. **Tracking Algorithm:** Various tracking algorithms can be employed to track the object over time. Some commonly used algorithms include:
    1. *Kalman Filter:* It uses a mathematical model to estimate the object's state and predict its future location based on motion dynamics.
    2. *Particle Filter:* Also known as the Monte Carlo filter, it uses a set of particles to represent the possible object locations and weights them based on their likelihood.
    3. *MeanShift:* It iteratively adjusts the position of the object based on the distribution of feature points, maximizing the similarity between the target and its surrounding area.
    4. *Correlation Filter:* It uses a learned correlation filter to track the object by maximizing the response in the filter's output.

5. **Update and Adaptation:** As new frames are processed, the tracking algorithm continually updates the object's location based on the detected features and motion estimation. The tracking algorithm may also adapt to changes in appearance, scale, or occlusion of the object.

6. **Occlusion Handling:** Object tracking algorithms often face challenges like occlusion, where the object of interest may be partially or completely obscured by other objects. Various techniques can be employed to handle occlusion, such as predicting the object's location based on motion or employing object re-detection methods.

7. **Termination Criteria:** Object tracking usually continues until a termination criterion is met. This criterion can be a specified number of frames, loss of tracking confidence, or reaching the end of the video sequence.
