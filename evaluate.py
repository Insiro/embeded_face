from os import path, makedirs
import numpy as np
import tensorflow as tf
import yaml
import json
from utils.util import load_data2
from models.model import build_face_model
from sklearn.metrics import classification_report
from dataclasses import dataclass, asdict


@dataclass
class EvaluateResult:
    precision: float
    recall: float
    acc: float


def get_values(labels, preds):
    repot = classification_report(labels, preds)
    recall = tf.keras.metrics.Recall()
    acc = tf.keras.metrics.Accuracy()
    precision = tf.keras.metrics.Precision()
    acc.update_state(labels, preds)
    recall.update_state(labels, preds)
    precision.update_state(labels, preds)
    return (
        EvaluateResult(
            float(precision.result()),
            float(recall.result()),
            float(acc.result()),
        ),
        repot,
    )


def evaluate(model, ds, as_dict=False):
    predictions = np.array([])
    labels = np.array([])
    for x, y in ds:
        ret = model.predict(x)
        predictions = np.concatenate([predictions, np.argmax(ret, axis=-1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
    # mat = multilabel_confusion_matrix(labels, predictions)
    value, report = get_values(labels, predictions)
    # ConfusionMatrixDisplay(confusionMatrix).plot(cmap="OrRd")
    report = classification_report(labels, predictions)
    recall = tf.keras.metrics.Recall()
    acc = tf.keras.metrics.Accuracy()
    precision = tf.keras.metrics.Precision()
    recall.update_state(labels, predictions)
    precision.update_state(labels, predictions)
    acc.update_state(labels, predictions)
    ret = {
        "recall": recall.result(),
        "precision": precision.result(),
        "acc": acc.result(),
    }
    print(report)
    if as_dict:
        return asdict(value)
    return value


def _main(coonfig):
    train_ds, test_ds, val_ds, _class_names = load_data2(config)
    model = build_face_model(100)
    model.load_weights("./output/out2/epoch172_loss2.90466_acc0.416")
    model.summary()
    value = evaluate(model, test_ds, as_dict=True)
    output_path = path.join(config['dir']["output"], "result")
    if not path.exists(output_path):
        makedirs(output_path)
    with open(path.join(output_path, "result.json"), "w") as f:
        json.dump(value, f)


if __name__ == "__main__":
    config = None
    with open("./config.yaml", "r") as cf:
        config = yaml.load(cf, Loader=yaml.Loader)
    with tf.device("/cpu:0"):
        _main(config)
