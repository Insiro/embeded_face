import tensorflow as tf


keras = tf.keras
from keras.applications import MobileNet
from keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Input,
    Flatten,
    Lambda,
)
from keras.models import Model
from .layers import ArcLayer, ArcMarginPenaltyLogists


class FaceMobile(Model):
    def __init__(self, num_classes, shape):
        super().__init__()
        self.mobilenet = MobileNet(
            weights="imagenet", include_top=False, input_shape=(shape[0], shape[1], 3)
        )
        self.mobilenet.trainable = False
        self.avgpool = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.dense1 = Dense(1024, activation="relu")
        self.dense2 = Dense(num_classes, activation="relu")
        self.norm = Lambda(
            lambda x: tf.math.l2_normalize(x, axis=-1), name="embeddings"
        )
        self.arc = ArcLayer(num_classes)

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.mobilenet(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.nrom(x)
        x = self.arc(x)
        return x


def build_face_model(num_calsses, shape=(480, 640)):
    inputs = Input(shape=(shape[0], shape[1], 3))
    mobilenet = MobileNet(
        weights="imagenet", include_top=False, input_shape=(shape[0], shape[1], 3)
    )
    mobilenet.trainable = False
    x = mobilenet(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(num_calsses, activation="relu")(x)
    x = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name="embeddings")(x)
    out = ArcLayer(num_calsses)(x)
    return Model(inputs=inputs, outputs=out)
