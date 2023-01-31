from os import path
from typing import Optional

import tensorflow as tf

keras = tf.keras
from keras.callbacks import Callback
from util import PathLoader, TrainingLog


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
            or ((self.test_loss > logs.test_loss) or (self.test_acc < logs.test_acc))
            or (
                (self.train_loss > logs.train_loss) or (self.train_acc < logs.train_acc)
            )
        ):
            self.test_loss = logs.test_loss
            self.test_acc = logs.test_acc
            self.train_loss = logs.train_loss
            self.train_acc = logs.train_acc
            filename = f"epoch{epoch}_loss{self.test_loss:.5f}_acc{logs.test_acc:.3f}"
            logs.model.save_weights(
                path.join(self.save_dir, filename), save_format="h5"
            )
            print("loss is decreased, file saved to {filename}")


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
        print("\tAcc : {logs.test_acc}\tLoss: {logs.test_loss}")

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
