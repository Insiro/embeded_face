from os import path
from os import mkdir
import tensorflow as tf

keras = tf.keras
from keras.callbacks import Callback


def convert(model):
    coverter = tf.lite.TFLiteConverter.from_keras_model(model)
    coverter.target_spec.supported_types = [tf.float32]
    coverter.optimizations = [tf.lite.Optimize.DEFAULT]
    return coverter.convert()


class SaveModelCallbacak(Callback):
    def __init__(self, base_dir: str = "./", prefix: str = "exp"):
        super().__init__()
        self.path = base_dir
        self.prefix = prefix
        self.save_dir = path.join(self.path, self.prefix)

    def on_train_begin(self, logs=None):
        newPath = self.save_dir
        folder_num = 1
        folder_name = ""
        while True:
            folder_name = f"{newPath}{folder_num}"
            folder_num += 1
            if not path.exists(folder_name):
                break
        mkdir(folder_name)
        self.save_dir = folder_name

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        filename = f"epoch{epoch}_loss{logs.test_loss}.h5"
        logs.model.save(path.join(self.save_dir, filename))


class SaveSummaryCallback(Callback):
    def __init__(self, log_path):
        super().__init__()
        self.path = log_path

    def on_train_begin(self, logs=None):
        self.writer = tf.summary.create_file_writer(path)

    @tf.function
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        with self.writer.as_default():
            tf.summary.scalar("test_loss", logs["test_loss"], step=epoch)
            tf.summary.scalar("train_loss", logs["train_loss"], step=epoch)
            tf.summary.scalar("test_acc", logs["test_acc"], step=epoch)
            tf.summary.scalar("train_acc", logs["train_acc"], step=epoch)
            tf.summary.scalar("lr", self.optimizer.lr, step=epoch)
        self.writer.flush()
