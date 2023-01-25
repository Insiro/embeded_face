import tensorflow as tf

keras = tf.keras
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from keras.applications import MobileNet

from metrics import ArcFace


class face_mobile(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.mobilenet = MobileNet(weights="imagenet", include_top=False)
        self.avgpool = GlobalAveragePooling2D()
        self.mobilenet.trainable = False
        self.batch_norm = BatchNormalization()
        self.dense1 = Dense(1024, activation="relu")
        self.dense2 = Dense(1024, activation="relu")
        # self.dense3 = Dense(512, activation="relu")
        # self.dense4 = Dense(num_classes, activation="softmax")
        self.arc = ArcFace(num_classes)

    def call(self, inputs, labels, training=False):
        y = labels
        x = self.mobilenet(inputs)
        x = self.avgpool(x)
        x = self.batch_norm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        # x = self.dense3(x)
        # x = self.dense4(x)
        x = self.arc([x, y])
        return x
