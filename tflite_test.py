from os import path

import numpy as np
import yaml
from cv2 import imread
from tensorflow import lite as tflite

from models.model import build_face_model
from utils.util import convert


def get_indxs(pwd):
    with open(pwd, "r") as f:
        lines = f.readlines()
        return lines


def load_model_save_tfl(config):
    model = build_face_model(config["classes"])
    model.load_weights(config["weight"])
    tflite_model = convert(model)
    tflite_path = path.join(config["dir"]["output"], "result", "converted_model.tflite")
    open(tflite_path, "wb").write(tflite_model)
    return tflite_path


class TfLiteModel:
    def __init__(self, tflite_path) -> None:
        self.interpeter = tflite.Interpreter(tflite_path)
        self.interpeter.allocate_tensors()
        self.input_details = self.interpeter.get_input_details()
        self.output_detial = self.interpeter.get_output_details()

    def predict(self, inputs):
        self.interpeter.set_tensor(self.input_details[0]["index"], inputs)
        self.interpeter.invoke()
        out = self.interpeter.get_tensor(self.output_detial[0]["index"])
        return out


def main(config):
    idx_list = get_indxs(path.join(config["dir"]["output"], "./classes.txt"))
    tflite_path = load_model_save_tfl(config)
    img = imread(
        "/data/01.datasets/FaceBank/Trained_Test/ID40/ID40-F-W0N1G_O.bmp"
    ).astype(np.float32)
    model = TfLiteModel(tflite_path)
    out = model.predict([img])[0]
    idx = np.argmax(out)
    print(out[idx], idx)
    thr = 0.85639226  # tpr 0.90745098,  fpr 0.90745098
    print(idx_list[idx])


if __name__ == "__main__":
    config = None
    with open("./config.yaml", "r") as cf:
        config = yaml.load(cf, Loader=yaml.Loader)
    main(config)
