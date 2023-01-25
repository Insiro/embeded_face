from os import path, mkdir, makedirs
import tensorflow as tf


def convert(model):
    coverter = tf.lite.TFLiteConverter.from_keras_model(model)
    coverter.target_spec.supported_types = [tf.float32]
    coverter.optimizations = [tf.lite.Optimize.DEFAULT]
    return coverter.convert()


class TrainingLog:
    def __init__(self, model) -> None:
        self.model = model
        self.test_loss = None
        self.train_loss = None
        self.test_acc = None
        self.train_acc = None
        self.lr = 0

    def update_test(self, acc, loss):
        self.test_acc = acc.result()
        self.test_loss = loss.result()

    def update_train(self, model, acc, loss, lr):
        self.model = model
        self.train_acc = acc.result()
        self.train_loss = loss.result()
        self.lr = lr


class PathLoader:
    def __init__(self, base_dir: str = "./", prefix: str = "exp"):
        self.path = base_dir
        self.prefix = prefix
        self.save_dir = None

    def __difine_path(self):
        if not path.exists(self.path):
            makedirs(self.path)
        newPath = path.join(self.path, self.prefix)
        folder_num = 1
        folder_name = ""
        while True:
            folder_name = f"{newPath}{folder_num}"
            folder_num += 1
            if not path.exists(folder_name):
                mkdir(folder_name)
                self.save_dir = folder_name
                return

    def get_save_path(self):
        if self.save_dir is None:
            self.__difine_path()
        return self.save_dir
