import tensorflow as tf
from tqdm import tqdm
from keras.callbacks import Callback


class AccMatrix:
    def __init__(self, train_loss, train_acc, test_loss, test_acc) -> None:
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.test_loss = test_loss
        self.test_acc = test_acc

    def reset_states(self):

        self.train_acc.reset_states()
        self.test_acc.reset_states()

    def get_train_str(self):
        return f"Accuracy: {self.train_acc.result() * 100}"

    def get_test_str(self):
        return f"Test Accuracy: {self.test_acc.result() * 100}"

    def __str__(self) -> str:
        return f"{self.get_train_str()}\t{self.get_train_str()}"


class ModelTrainer:
    def __init__(
        self,
        model,
        logger,
        loss_function,
        optimizer,
        acc_matrix: AccMatrix,
    ) -> None:
        self.model = model
        self.logger = logger
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.acc_matrix = acc_matrix
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.test_loss = tf.keras.metrics.Mean(name="train_loss")
        self.callback: list[Callback] = []

    def add_callback(self, callback: Callback):
        self.callback.append(callback)

    @tf.function
    def test_step(self, image, labels):
        pred = self.model(image, training=False)
        loss = self.loss_function(labels, pred)
        self.test_loss(loss)
        self.acc_matrix.test_acc(labels, pred)

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as grad:
            pred = self.model(images, training=True)
            loss = self.loss_function(labels, pred)
        gradinents = grad.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradinents, self.model.trainable_variables))
        self.train_loss(loss)
        self.acc_matrix.train_acc(labels, pred)

    def reset_states(self):
        self.acc_matrix.reset_states()
        self.train_loss.reset_states()
        self.test_loss.reset_states()

    def train(self, epochs, train_set, test_set):
        for callback in self.callback:
            callback.on_epoch_begin(self)
        for epoch in range(epochs):
            self.reset_states()
            for image, labels in tqdm(train_set, desc=f"train {epoch}epoch"):
                self.train_step(image, labels)
            print(
                f"Acc : {self.acc_matrix.get_train_str()}\tLoss: {self.train_loss.result()}"
            )
            for image, labels in tqdm(test_set, desc=f"test {epoch}epoch"):
                self.test_step(image, labels)

            print(
                f"Acc : {self.acc_matrix.get_test_str()}\tLoss: {self.test_loss.result()}",
            )
            logs = {
                "model": self.model,
                "test_loss": self.test_loss.result(),
                "train_loss": self.train_loss.result(),
                "test_acc": self.acc_matrix.test_acc.result(),
                "train_acc": self.acc_matrix.train_acc.result(),
            }
            for callback in self.callback:
                callback.on_epoch_end(self, logs=logs)
