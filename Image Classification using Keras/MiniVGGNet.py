import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the input shape and channel dimension, assuming tensorflow/channels-last ordering
        inputShape = (height, width, depth)
        chanDim = -1

        # define the model input
        inputs = layers.Input(shape = inputShape)

        # first (conv => RELU) * 2 => POOL layer set
        x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Conv2D(32, (3, 3), padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # second (conv => RELU) * 2 => POOL layer set
        x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Conv2D(32, (3, 3), padding="same")(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # first (and only) set of FC => RELU layers
        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Dropout(0.5)(x)

        # softmax classifier
        x = layers.Dense(classes)(x)
        x = layers.Activation("softmax")(x)

        model = models.Model(inputs, x, name="minivggnet")
        return model