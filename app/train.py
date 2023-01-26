from os import path

import tensorflow as tf
import yaml
from numba import cuda

keras = tf.keras
from callbacks import SaveModelCallbacak, SaveSummaryCallback
from metrics import ArcLoss
from model import build_face_model
from trainer import AccMatrix, ModelTrainer
from util import PathLoader, convert, load_data


def main(config):
    pathLoader = PathLoader(config["output_dir"])
    train_ds, test_ds, val_ds, _class_names = load_data(config, "test")

    callbacks = [
        SaveModelCallbacak(path_loader=pathLoader),
        SaveSummaryCallback(path_loader=pathLoader),
    ]

    lr_scheduler = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001, first_decay_steps=50, t_mul=2, m_mul=0.9
    )
    # model = FaceMobile(510, shape=(480, 640))
    model = build_face_model(510, shape=config["shape"])
    loss_function = ArcLoss()
    optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler)

    train_loss = keras.metrics.Mean(name="train_loss")
    train_acc = keras.metrics.CategoricalAccuracy(name="train_acc")
    test_loss = keras.metrics.Mean(name="test_loss")
    test_acc = keras.metrics.CategoricalAccuracy(name="test_acc")
    trainer = ModelTrainer(
        model,
        None,
        loss_function=loss_function,
        optimizer=optimizer,
        acc_matrix=AccMatrix(
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        ),
    )
    trainer.add_callback(callbacks)
    trainer.train(config["epochs"], train_ds, test_ds)

    image, labels = tuple(zip(*val_ds))
    result = model.evaluate(image, labels)
    with open(path.join(pathLoader.get_save_path(), "result.txt"), "w") as ff:
        ff.write(result)

    model.save(path.join(pathLoader.get_save_path(), "./fianl.h5"))
    open(path.join(pathLoader.get_save_path(), "./test.tflite"), "wb").write(
        convert(model)
    )


if __name__ == "__main__":
    cuda.get_current_device().reset()
    with open("./config.yaml", "r") as cf:
        config = yaml.load(cf, Loader=yaml.Loader)

    with tf.device("/gpu:0"):
        main(config)
