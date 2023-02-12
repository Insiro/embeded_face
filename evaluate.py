import json
from dataclasses import asdict, dataclass
from os import makedirs, path

import numpy as np
import tensorflow as tf
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import auc, classification_report, roc_curve
from tqdm import tqdm

from models.model import build_face_model
from utils.util import load_data2

import pandas as pd


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


def gen_roc(y_test, y_probs, outdir):
    num_classes = 100
    plt.title("ROC curve for multi-class classifier")
    # Compute the ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], ths = roc_curve(y_test[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve for class {} (area = {:.2f})".format(i, roc_auc[i]),
        )
    # Plot the ROC curves for each class
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    df = pd.DataFrame()
    df["fpr"], df["tpr"], df["ths"] = roc_curve(y_test.ravel(), y_probs.ravel())

    df.to_csv(f"{path.join(outdir,'./TprFprThr.csv') }")
    plt.savefig(f"{path.join(outdir,'./roc.png') }")


def evaluate(model, ds, outdir, as_dict=False):
    predictions = np.array([])
    labels = np.array([])
    label_onehot = None
    prob = None
    for x, y in tqdm(ds):
        ret = model.predict(x, verbose=False)
        predictions = np.concatenate([predictions, np.argmax(ret, axis=-1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
        prob = ret.copy() if prob is None else np.vstack([prob, ret])
        label_onehot = y if label_onehot is None else np.vstack([label_onehot, y])
    gen_roc(label_onehot, prob, outdir)
    value, report = get_values(labels, predictions)
    print(report)
    if as_dict:
        return asdict(value)
    return value


def _main(config):
    _, test_ds, _, _ = load_data2(config)
    model = build_face_model(config["classes"])
    model.load_weights(config["weight"])
    model.summary()
    output_path = path.join(config["dir"]["output"], "result")
    value = evaluate(model, test_ds, output_path, as_dict=True)
    if not path.exists(output_path):
        makedirs(output_path)
    with open(path.join(output_path, "result.json"), "w") as f:
        json.dump(value, f)


if __name__ == "__main__":
    config = None
    with open("./config.yaml", "r") as cf:
        config = yaml.load(cf, Loader=yaml.Loader)
    with tf.device("/gpu:0"):
        _main(config)
