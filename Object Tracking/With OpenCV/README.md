In this repository, I will tell you about eight separate object tracking algorithm built right into OpenCV.

## 8 OpenCV Object Tracking Implementations ##
I've included a brief highlight of each object tracker below:
1. **BOOSTING Tracker:** Based on the same algorithm used to power the machine learning behind Haar cascades (AdaBoost), but like Haar cascades, is over a decade old. This tracker is slow and doesn’t work very well. Interesting only for legacy reasons and comparing other algorithms. (minimum OpenCV 3.0.0)
2. **MIL Tracker:** Better accuracy than BOOSTING tracker but does a poor job of reporting failure. (minimum OpenCV 3.0.0)
3. **KCF Tracker:** Kernelized Correlation Filters. Faster than BOOSTING and MIL. Similar to MIL and KCF, does not handle full occlusion well. (minimum OpenCV 3.1.0)
4. **CSRT Tracker:** Discriminative Correlation Filter (with Channel and Spatial Reliability). Tends to be more accurate than KCF but slightly slower. (minimum OpenCV 3.4.2)
5. **MedianFlow Tracker:** Does a nice job reporting failures; however, if there is too large of a jump in motion, such as fast moving objects, or objects that change quickly in their appearance, the model will fail. (minimum OpenCV 3.0.0)
6. **TLD Tracker:** I’m not sure if there is a problem with the OpenCV implementation of the TLD tracker or the actual algorithm itself, but the TLD tracker was incredibly prone to false-positives. I do not recommend using this OpenCV object tracker. (minimum OpenCV 3.0.0)
7. **MOSSE Tracker:** Very, very fast. Not as accurate as CSRT or KCF but a good choice if you need pure speed. (minimum OpenCV 3.4.1)
8. **GOTURN Tracker:** The only deep learning-based object detector included in OpenCV. It requires additional model files to run (will not be covered in this repository). GOTURN changed the way we apply Deep Learning to the problem of tracking by learning the motion of an object in an offline manner. The GOTURN model is trained on thousands of video sequences and does not need to perform any learning at runtime.

**My suggestion is to:**
* Use CSRT when you need **higher object tracking accuracy and can tolerate slower FPS throughput**
* Use KCF when you need **faster FPS throughput** but can **handle slightly lower object tracking accuracy**
* Use **MOSSE when you need pure speed**

Satya Mallick also provides some additional information on these object trackers [in his article as well](https://learnopencv.com/object-tracking-using-opencv-cpp-python/).




