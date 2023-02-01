import logging
from os import path
from typing import Optional

import numpy as np
import tensorflow as tf

keras = tf.keras
from keras.callbacks import Callback

from utils.util import PathLoader, TrainingLog


class SaveModelCallbacak(Callback):
    def __init__(self, path_loader: PathLoader):
        super().__init__()
        self.save_dir = path_loader.get_save_path()
        self.train_loss = None
        self.train_acc = None
        self.test_acc = None
        self.test_loss = None

    def on_epoch_end(self, epoch, logs: Optional[TrainingLog] = None):
        if logs is None:
            return
        if (
            (self.train_loss is None)
            or (self.test_loss > logs.test_loss)
            or (self.test_acc < logs.test_acc)
            or (self.train_loss > logs.train_loss)
            or (self.train_acc < logs.train_acc)
        ):
            self.test_loss = logs.test_loss
            self.test_acc = logs.test_acc
            self.train_loss = logs.train_loss
            self.train_acc = logs.train_acc
            filename = f"epoch{epoch}_loss{self.test_loss:.5f}_acc{logs.test_acc:.3f}"
            logs.model.save_weights(
                path.join(self.save_dir, filename), save_format="h5"
            )
            logging.info(f"loss is decreased, file saved to {filename}")


class SaveSummaryCallback(Callback):
    def __init__(self, path_loader: PathLoader, save_logs=True):
        super().__init__()
        self.save_dir = path_loader.get_save_path()
        self.save_logs = save_logs

    def on_train_begin(self, logs: Optional[TrainingLog] = None):
        if self.save_logs == True:
            self.writer = tf.summary.create_file_writer(self.save_dir)

    def on_train_end(self, logs: Optional[TrainingLog] = None):
        if logs is None:
            return
        print(
            f"\tAcc : {logs.train_acc}\tLoss: {logs.train_loss}\t lr : {float(logs.lr)}"
        )

    def on_test_end(self, logs: Optional[TrainingLog] = None):
        if logs is None:
            return
        print(f"\tAcc : {logs.test_acc}\tLoss: {logs.test_loss}")

    @tf.function
    def on_epoch_end(self, epoch, logs: Optional[TrainingLog] = None):
        if logs is None or self.save_logs == False:
            return
        with self.writer.as_default():
            tf.summary.scalar("test_loss", logs.test_loss, step=epoch)
            tf.summary.scalar("train_loss", logs.train_loss, step=epoch)
            tf.summary.scalar("test_acc", logs.test_acc, step=epoch)
            tf.summary.scalar("train_acc", logs.train_acc, step=epoch)
            tf.summary.scalar("lr", logs.lr, step=epoch)
        self.writer.flush()


class EarlyStop(Callback):
    class Metrix:
        def reset(self) -> None:
            self.acc = np.Inf
            self.loss = -np.Inf

    def __init__(
        self,
        path_loader: PathLoader,
        start_epoch=0,
        save_best=False,
        patience=50,
        min_delta=0,
    ):
        super().__init__()
        self.start_epoch = start_epoch
        self.save_best = save_best
        self.best_metrix = self.Metrix()
        self.save_dir = path_loader.get_save_path()
        self.patience = patience
        self.loss_delta = -min_delta
        self.acc_delta = min_delta
        self.moniter = "loss"

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_metrix.reset()
        self.best_weight = None
        self.best_epoch = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs: Optional[TrainingLog] = None):
        if logs is None or epoch < self.start_epoch:
            return
        if self.save_best and self.best_weight is None:
            self.best_weight = logs.model.get_weights()
        self.wait += 1
        if self._is_imporved(logs):
            self._update_best(logs)
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            logs.stop_train = True
            if self.save_best and self.best_weight is not None:
                filename = f"best_loss{self.best_metrix.loss:.5f}_acc{self.best_metrix.acc:.3f}"
                logging.info(f"Restoring model weights to {filename}")
                logs.model.set_weights(self.best_weight)
                logs.model.save_weights(
                    path.join(self.save_dir, filename), save_format="h5"
                )

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch}: early stopped")
        return super().on_train_end(logs)

    def _update_best(self, logs: Optional[TrainingLog]):
        if logs is None:
            return
        self.best_acc = logs.test_acc
        self.best_loss = logs.test_loss
        if self.save_best:
            self.best_weight = logs.model.get_weights()

    def _is_imporved(self, logs: Optional[TrainingLog])->bool:
        if logs is None:
            return False
        increse_acc = (logs.test_acc - self.acc_delta) > self.best_metrix.acc
        decrease_loss = (logs.test_loss - self.loss_delta) < self.best_metrix.loss
        if self.moniter == "loss":
            return decrease_loss
        if self.moniter == "acc":
            return increse_acc
        if self.moniter == "any":
            return increse_acc or decrease_loss
        if self.moniter == "all":
            return increse_acc and decrease_loss
        return decrease_loss