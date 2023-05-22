# Face Detection #
A face detection system can be defined as a system or technology that can identify and identify individuals from a photo or video frame from a source video. There are several methods by which face recognition systems usually operate, by comparing the facial characteristics extracted from the given image with the faces already stored in a dataset. Here we'll explore how face detection and recognition work using different techniques like Haar cascade, DNN Face Detector in OpenCV, Dlib HOG-based, and Dlib CNN-based.

## Face Detection Techniques ##
There are several face-detection techniques, but in this paper, we will be focusing on the four most popular face-detection techniques namely:
1. Haar Cascade
2. DLib HOG-based
3. DLib CNN-based
4. DNN with OpenCV

#### Haar Cascade ####
  * It's super quick and it pulls more functions from pictures much like the basic CNN.
  * The Haar Cascade model size is tiny (930 KB).
  * It works almost real-time on a CPU, has a simple architecture and can detect faces at different scales.
  * The 4 main steps in the Haar cascade are Haar features extraction, Integral Images concept, Adaboost training and cascading of the classifier. Haar- function     stores images in frames, where many pixels are in a row. Each box then analyzed and created a difference in value that showed the dark and luminous areas. Those values are used for image processing. You can then pick the best functionality by Adaboost. This cuts the initial functionality of 160000+ to 6000. However, it takes a lot of time to apply all these functions in a sliding pane. They have thus added a Cascade of Classifiers, which combines functions. If a first stage window stalls, these residual functions are not processed in the cascade. If the following function passes, it is checked and replicated in the same process. If the following function passes, it is checked and replicated in the same process. If a window can pass all features, it is known as a facial region.
  * Haar Cascades uses the Adaboost learning algorithm that selects a small number of important features from a wide collection to give classifiers an efficient result.

#### DLib Hog-based ####
  * Dlib is a toolkit in C++ which contains algorithms for machine learning, which solve problems in the real world. 
  * Dlib works to supply the frontal face detector with functions taken from the Histogram of Oriented Gradients (HOG), which are then transferred into an SVM. The distribution of gradient directions is used as characteristics in the HOG function descriptor
  * The concept behind HOG is to extract features into a vector and to feed it into a classifying algorithm such as, for instance, a vector supporting machine that will determine whether a face is present or not in a field. The characteristics extracted are the distribution of gradient (oriented gradient) directions of the image. Gradients around edges and corners are usually wide which allows us to detect these areas.
  * The following are the steps involved:
      a. Pre-processing
      b. Compute the Gradient Images
      c. HOG computation
      d. Block Normalization
  
#### DLib CNN-based ####
  * The Convolutional neural network (CNN) is a neural feed system that is used primarily for computer vision. They give a dense neural network component as well as an artificial picture pre-treatment.
  * CNN's are special kinds of neural networks for grid-like data analysis.
  * In previous methods, much of the work involved selecting filters to establish the characteristics such that as much detail could be extracted from the image as possible. This work can now be automated with increasing profound understanding and greater computational skills.
  * The CNNs are called since the original image data is combined with a series of filters. The number of filters to be applied is the parameter to be selected and the filtration dimension. The filter dimension is known as the step length. Typical phase values range from 2 to 5. In this particular case, the output of the CNN is a binary classification that takes value 1 when a face exists, otherwise value 0. A paper on Max-Margin Object Detection (MMOD) is also implemented for improved performance.
  * This model works with various facial orientations and is occlusion sturdy and its training method really fast. But is very slow and cannot detect smaller faces since they are specialized in the size of 80 to 80 faces. You must then ensure that the face size of the submission is larger than that. However, for smaller faces, you should train your own facial detector.

#### DNN with OpenCV ####
  * It's a Caffe model built upon the Single Shot-Multibox Detector (SSD) and its foundation is ResNet-10. It was launched in its deep neural network (DNN) module after OpenCV 3.3.
  *  A quantified version of TensorFlow is also available, but we are going to use the Caffe model. This model is very reliable, works on CPU in real-time, works well on various facial orientations, works even with significant occlusion and can even detect faces of different sizes.
  *  OpenCV provides 2 models for this face detector.
      * Floating-point 16 version of the original Caffe implementation (5.4 MB)
      * The 8-bit quantized version using TensorFlow (2.7 MB)

## Comparative Analysis of face detection techniques ##
| SN. |        CRITERIA            |      FIRST             |       SECOND        |       THIRD          |       FOURTH       |         
| --- | -------------------------  | ---------------------- | ------------------- | -------------------- | ------------------ |
|  1  | Accuracy                   | OpenCV - DNN           | Haar Cascade        | Dlib - CNN           | Dlib - HOG         |
|  2  | Speed (FPS)                | Haar Cascade (~12fps)  | Dlib – HOG (~7fps)  | OpenCV – DNN (~5fps) | Dlib – CNN (~3fps) |
|  3  | Best Result with Occlusion | OpenCV - DNN           | Dlib - CNN          | Dlib - HOG           | Haar Cascade       |
|  4  | Lighting                   | Haar Cascade           | Dlib - HOG          | Dlib - CNN           | OpenCV - DNN       |
|  5  | Scale                      | OpenCV - DNN           | Haar Cascade        | Dlib - HOG           | Dlib - CNN         |
|  6  | Face Angle                 | OpenCV - DNN           | Dlib - CNN          | Dlib - HOG           | Haar Cascade       |
|  7  | Motion                     | OpenCV - DNN           | Haar Cascade        | Dlib - CNN           | Dlib - HOG         |

