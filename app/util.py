import tensorflow as tf
from os import path
from os import listdir, mkdir


def convert(model):
    coverter = tf.lite.TFLiteConverter.from_keras_model(model)
    coverter.target_spec.supported_types = [tf.float32]
    coverter.optimizations = [tf.lite.Optimize.DEFAULT]
    return coverter.convert()


class SaveModelCallbacak(tf.keras.callbacks.Callback):
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

    def on_epoch_end(self, logs=None):
        filename = f"epoch{logs.epoch}_loss{logs.test_loss}.h5"
        logs.model.save(path.join(self.save_dir, filename))
