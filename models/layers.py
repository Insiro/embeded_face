import tensorflow as tf

keras = tf.keras
from keras.layers import Layer


class ArcLayer(Layer):
    def __init__(self, n_classes, **kwargs):
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
