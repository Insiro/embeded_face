from typing import Union

import tensorflow as tf
from tqdm import tqdm

keras = tf.keras

from keras.callbacks import Callback
from utils.util import TrainingLog


class AccMatrix:
    def __init__(self, train_loss, train_acc, test_loss, test_acc) -> None:
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.test_loss = test_loss
        self.test_acc = test_acc

    @tf.function
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
        self.model: tf.keras.Model = model
        self.logger = logger
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.acc_matrix = acc_matrix
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.test_loss = tf.keras.metrics.Mean(name="train_loss")
        self.callback: list[Callback] = []

    def add_callback(self, callback: Union[Callback, list[Callback]]):
        if callback is Callback:
            self.callback.append(callback)
        else:
            self.callback.extend(callback)

    @tf.function
    def test_step(self, images, labels):
        pred = self.model(images)
        loss = self.loss_function(labels, pred) + sum(self.model.losses)
        self.test_loss(loss)
        self.acc_matrix.test_acc(labels, pred)

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as grad:
            pred = self.model(images)
            loss = self.loss_function(labels, pred) + sum(self.model.losses)
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
            callback.on_train_begin(self)
        logs = TrainingLog(self.model)

        for epoch in range(epochs):
            self.reset_states()
            for image, labels in tqdm(train_set, desc=f"train {epoch} epoch"):
                self.train_step(image, labels)
            logs.update_train(
                self.acc_matrix.train_acc,
                self.train_loss,
                self.optimizer.lr,
            )
            for callback in self.callback:
                callback.on_train_end(logs=logs)

            for image, labels in tqdm(test_set, desc=f"test  {epoch} epoch"):
                self.test_step(image, labels)
            logs.update_test(self.acc_matrix.test_acc, self.test_loss)
            for callback in self.callback:
                callback.on_test_end(logs=logs)

            for callback in self.callback:
                callback.on_epoch_end(epoch, logs=logs)
            if logs.stop_train:
                break
        for callback in self.callback:
            callback.on_train_end(logs=logs)
