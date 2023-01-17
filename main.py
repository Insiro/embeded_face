import tensorflow as tf

keras = tf.keras
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from tqdm import tqdm

EPOCHS = 100000
BATCH_SIZE = 5
DATA_DIR = "/data/01.datasets/MorphedImageBank"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(480, 600),
    batch_size=BATCH_SIZE,
)
class_names = train_ds.class_names


class face_mobile(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.mobilenet = MobileNet(weights="imagenet", include_top=False)
        self.avgpool = GlobalAveragePooling2D()
        self.mobilenet.trainable = False
        self.dense1 = Dense(1024, activation="relu")
        self.dense2 = Dense(1024, activation="relu")
        self.dense3 = Dense(512, activation="relu")
        self.dense4 = Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.mobilenet(inputs)
        x = self.avgpool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


class ModelTrainer:
    def __init__(self, model, data, logger, loss_function, optimizer) -> None:
        self.model = model
        self.data = data
        self.logger = logger
        self.loss_function = loss_function
        self.optimizer = optimizer

    @tf.function
    def test_step(self, image, labels):
        pred = model(image, training=False)
        loss = self.loss_function(labels, pred)
        test_loss(loss)
        test_acc(labels, pred)

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as grad:
            pred = self.model(images, training=True)
            loss = self.loss_function(labels, pred)
        gradinents = grad.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradinents, self.model.trainable_variables))
        train_loss(loss)
        train_acc(labels, pred)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss.reset_states()
            train_acc.reset_states()
            test_loss.reset_states()
            test_acc.reset_states()
            for image, labels in tqdm(train_ds):

                self.train_step(image, labels)
            for image, labels in tqdm(train_ds):
                self.test_step(image, labels)

            print(
                f"Epoch {epoch + 1}, "
                f"Loss: {train_loss.result()}, "
                f"Accuracy: {train_acc.result() * 100}, "
                f"Test Loss: {test_loss.result()}, "
                f"Test Accuracy: {test_acc.result() * 100}"
            )


model = None
with tf.device("/gpu:0"):
    model = face_mobile(510)
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam()

    train_loss = keras.metrics.Mean(name="train_loss")
    train_acc = keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    test_loss = keras.metrics.Mean(name="test_loss")
    test_acc = keras.metrics.SparseCategoricalAccuracy(name="test_acc")
    trainer = ModelTrainer(
        model, None, None, loss_function=loss_function, optimizer=optimizer
    )
    trainer.train(EPOCHS)


def convert(model):
    coverter = tf.lite.TFLiteConverter.from_keras_model(model)
    coverter.target_spec.supported_types = [tf.float32]
    coverter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = coverter.convert()
    open("./test.tflite", "wb").write(tflite_model)


model.save("./output.h5")
convert(model)
