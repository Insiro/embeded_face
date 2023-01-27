from os import path, makedirs
import numpy as np
import tensorflow as tf
import yaml
import json
from util import load_data2
from model import build_face_model
from sklearn.metrics import confusion_matrix, classification_report


def get_values(labels, preds):
    confusionMatrix = confusion_matrix(labels, preds)
    repot = classification_report(labels, preds)
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    recall.update_state(labels, preds)
    precision.update_state(labels, preds)

    return repot, recall.result(), precision.result()


with tf.device("/cpu:0"):
    config = None
    with open("./config.yaml", "r") as cf:
        config = yaml.load(cf, Loader=yaml.Loader)

    train_ds, test_ds, val_ds, _class_names = load_data2(config)
    model = build_face_model(100)
    model.load_weights("./output/out2/epoch172_loss2.90466_acc0.416")
    model.summary()

    predictions = np.array([])
    labels = np.array([])
    for x, y in test_ds:
        ret = model.predict(x)
        predictions = np.concatenate([predictions, np.argmax(ret, axis=-1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
    # mat = multilabel_confusion_matrix(labels, predictions)
    report, recall, precision = get_values(labels, predictions)
    # ConfusionMatrixDisplay(confusionMatrix).plot(cmap="OrRd")
    test_ret = {
        "recall": float(recall),
        "precision": float(precision),
        "report": report,
    }
    print(report)
    predictions = np.array([])
    labels = np.array([])
    for x, y in train_ds:
        ret = model.predict(x)
        predictions = np.concatenate([predictions, np.argmax(ret, axis=-1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
    report, recall, precision = get_values(labels, predictions)
    train_ret = {
        "recall": float(recall),
        "precision": float(precision),
        "report": report,
    }
    print(report)
    output_path = path.join(config["output_dir"], "result")
    if not path.exists(output_path):
        makedirs(output_path)
    with open(path.join(output_path, "result.json"), "w") as f:
        json.dump({"train": train_ret, "test": test_ret}, f)
