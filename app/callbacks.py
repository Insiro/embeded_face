from typing import Optional
from os import path
import tensorflow as tf

keras = tf.keras
from keras.callbacks import Callback
from util import PathLoader, TrainingLog


class SaveModelCallbacak(Callback):
    def __init__(self, path_loader: PathLoader):
        super().__init__()
        self.save_dir = path_loader.get_save_path()
        self.loss = None

    def on_epoch_end(self, epoch, logs: Optional[TrainingLog] = None):
        if logs is None:
            return
        if self.loss is None or self.loss > logs.test_loss:
            self.loss = logs.test_loss
            filename = f"epoch{epoch}_loss{self.loss}.h5"
            logs.model.save(path.join(self.save_dir, filename))


class SaveSummaryCallback(Callback):
    def __init__(self, path_loader: PathLoader):
        super().__init__()
        self.save_dir = path_loader.get_save_path()

    def on_train_begin(self, logs: Optional[TrainingLog] = None):
        if self.save_logs == True:
            self.writer = tf.summary.create_file_writer(self.path)

    def on_train_end(self, logs: Optional[TrainingLog] = None):
        if logs is None:
            return
        print(f"Acc : {logs.train_acc}\tLoss: {logs.train_loss}")

    def on_test_end(self, logs: Optional[TrainingLog] = None):
        if logs is None:
            return
        print(f"Acc : {logs.test_acc}\tLoss: {logs.test_loss}")

    @tf.function
    def on_epoch_end(self, epoch, logs: Optional[TrainingLog] = None):
        if logs is None or self.save_logs == False:
            return
        with self.writer.as_default():
            tf.summary.scalar("test_loss", logs.test_loss, step=epoch)
            tf.summary.scalar("train_loss", logs.train_loss, step=epoch)
            tf.summary.scalar("test_acc", logs.test_acc, step=epoch)
            tf.summary.scalar("train_acc", logs.train_acc, step=epoch)
            tf.summary.scalar("lr", self.optimizer.lr, step=epoch)
        self.writer.flush()
