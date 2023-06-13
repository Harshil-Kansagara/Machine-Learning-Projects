import matplotlib
matplotlib.use("Agg")

from MiniVGGNet import MiniVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
# from tensorflow.keras import optimizers
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot_tf.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# load the training and testing data, then scale it into the range [0,1]
print("[INFO] loading CIFAR-10 data....")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# initialize the initial learning rate, total number of epochs to train for, and batch size
learning_rate = 0.01
epochs = 30
batch_size = 32

# initialize the optimizer and model
print("[INFO] compiling model.....")
# opt = optimizers.SGD(lr = learning_rate, decay = learning_rate / epochs)
model = MiniVGGNet.build(width = 32, height=32, depth=3, classes=len(labelNames))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(epochs))
history = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batch_size, epochs=epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])