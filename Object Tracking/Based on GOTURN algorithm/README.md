# GOTURN #
* GOTURN stands for Generic Object Tracking Using Regression Networks, is a Deep Learning based tracking algorithm.
* Most tracking algorithms are trained in an online manner. In other words, the tracking algorithm learns the appearance of the object it is tracking at runtime. Therefore, many real-time trackers rely on online learning algorithms that are typically much faster than a Deep Learning based solution.
* GOTURN changed the way we apply Deep Learning to the problem of tracking by learning the motion of an object in an offline manner. The GOTURN model is trained on thousands of video sequences and does not need to perform any learning at runtime.

### How does GOTURN work? ###
![goturn-inputs-ouputs](https://github.com/Harshil-Kansagara/Machine-Learning-Projects/assets/35835271/39b2178d-f4cc-4dfa-9963-6ed94802be4b)

As shown in Fig, GOTURN is trained using a pair of cropped frames from the thousands of videos

In first frame (also referred to as the previous frame), the location of the object is known, and the frame is cropped to two times the size of the bounding box around the object. The object in the first cropped frame is always centered.

The location of the object in the second frame (also referred to as the current frame) needs to be predicted. The bounding box used to crop the first frame is also used to crop the second frame. Because the object might have moved, the object is not centered in the second frame.

A Convolutional Neural Network (CNN) is trained to predict the location of the bounding box in the second frame.

### GOTURN Architecture ###
![GOTURN-architecture](https://github.com/Harshil-Kansagara/Machine-Learning-Projects/assets/35835271/032ead72-6a90-4f41-8618-ee7a0ae542c0)

Figure shows the architecture of GOTURN.

As mentioned before, it takes two cropped frame as input. Notice, the previous frame, shown at the bottom, is centered and our goal is the find the bounding box for the currrent frame shown on the top.

Both frames pass through a bank of convolutional layers. The layers are simply the first five convolutional layers of the CaffeNet architecture. The outputs of these convolutional layers (i.e. the pool5 features) are concatenated into a single vector of length 4096. This vector is input to 3 fully connected layers. The last fully connected layer is finally connected to the output layer containing 4 nodes representing the top and bottom points of the bounding box.

### Download GOTURN model files ###
You can download the GOTURN caffemodel and prototxt files located at [this link](https://github.com/spmallick/goturn-files). The model file is split into 4 files which will need to be merged before unzipping

