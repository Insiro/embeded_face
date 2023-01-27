import tensorflow as tf

keras = tf.keras
from keras.applications import MobileNet
from keras.layers import (
    BatchNormalization,
    Dense,
    GlobalAveragePooling2D,
    Layer,
    Input,
    Softmax,
)
from keras.models import Model


class FaceMobile(Model):
    def __init__(self, num_classes, shape):
        super().__init__()
        self.mobilenet = MobileNet(
            weights="imagenet", include_top=False, input_shape=(shape[0], shape[1], 3)
        )
        self.avgpool = GlobalAveragePooling2D()
        self.mobilenet.trainable = False
        self.batch_norm = BatchNormalization()
        self.dense1 = Dense(1024, activation="relu")
        self.dense2 = Dense(512, activation="relu")
        self.arc = ArcLayer(num_classes)

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.mobilenet(x)
        x = self.avgpool(x)
        x = self.batch_norm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.arc(x)
        return x


class ArcLayer(Layer):
    def __init__(self, n_classes, kernel_regularizer=None, **kwargs):
        super(ArcLayer, self).__init__(**kwargs)
        self.n_classes = n_classes

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=[input_shape[-1], self.n_classes],
            dtype=tf.float32,
            initializer=keras.initializers.HeNormal(),
            trainable=True,
            name="kernel",
        )
        self.built = True

    @tf.function
    def call(self, inputs):
        weights = tf.nn.l2_normalize(self.kernel, axis=0)
        return tf.matmul(inputs, weights)


def build_face_model(num_calsses, shape=(480, 640)):
    print("shape", shape)
    inputs = Input(shape=(shape[0], shape[1], 3))
    mobilenet = MobileNet(
        weights="imagenet", include_top=False, input_shape=(shape[0], shape[1], 3)
    )
    mobilenet.trainable = False
    x = mobilenet(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    out = ArcLayer(num_calsses)(x)
    x = Softmax()(x)
    return Model(inputs=inputs, outputs=out)
