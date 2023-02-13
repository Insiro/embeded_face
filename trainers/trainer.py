from typing import Union

import tensorflow as tf
from tqdm import tqdm

keras = tf.keras

from keras.callbacks import Callback
from utils.util import TrainingLog

from .base import AccMatrix, TrainerBase


class ModelTrainer(TrainerBase):
    def __init__(
        self, model, logger, loss_function, optimizer, acc_matrix: AccMatrix
    ) -> None:
        super().__init__(model, logger, loss_function, optimizer, acc_matrix)

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
