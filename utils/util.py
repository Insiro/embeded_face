from os import makedirs, mkdir, path

import tensorflow as tf


def convert(model):
    coverter = tf.lite.TFLiteConverter.from_keras_model(model)
    coverter.target_spec.supported_types = [tf.float32]
    coverter.optimizations = [tf.lite.Optimize.DEFAULT]
    return coverter.convert()


def load_data(config, dir="data", split_val=False):
    data_path = config["dir"][dir]
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=config["valid_split"],
        subset="training",
        shuffle=False,
        label_mode="categorical",
        seed=123,
        image_size=config["shape"],
        batch_size=config["batch_size"],
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=config["valid_split"],
        subset="validation",
        shuffle=False,
        label_mode="categorical",
        seed=123,
        image_size=config["shape"],
        batch_size=config["batch_size"],
    )
    test_ds = None
    if split_val:
        val_batches = tf.data.experimental.cardinality(val_ds)

        test_ds = val_ds.take((2 * val_batches) // 2)
        val_ds = val_ds.skip((2 * val_batches) // 2)
    class_names = train_ds.class_names
    return train_ds, val_ds, test_ds, class_names


def load_data2(config):
    dir = config["dir"]
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dir["data"],
        shuffle=False,
        label_mode="categorical",
        seed=123,
        image_size=config["shape"],
        batch_size=config["batch_size"],
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dir["facebankTest"],
        shuffle=False,
        label_mode="categorical",
        seed=123,
        image_size=config["shape"],
        batch_size=config["batch_size"],
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, None, class_names


class TrainingLog:
    def __init__(self, model) -> None:
        self.model: tf.keras.Model = model
        self.test_loss = None
        self.train_loss = None
        self.test_acc = None
        self.train_acc = None
        self.lr = 0
        self.stop_train = False

    def update_test(self, acc, loss):
        self.test_acc = acc.result()
        self.test_loss = loss.result()

    def update_train(self, acc, loss, lr):
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
                print(f"output_dir : {self.save_dir}")
                return

    def get_save_path(self):
        if self.save_dir is None:
            self.__difine_path()
        return self.save_dir
