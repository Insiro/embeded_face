from os import listdir, path

import numpy as np
import tensorflow as tf
import yaml
from cv2 import imread
from keras import Model, Sequential
from tqdm import tqdm

from models.model import build_face_model
from predict import load_embeds, cos_sim
from sklearn.metrics import classification_report


def euclid_dis(x, y):
    return np.sum(np.square(x - y))


def dist(x, y):
    dist = tf.math.squared_difference(x, y)
    return np.sum(dist)


def dot_dist(x, y):
    dist = np.dot(x, y)

    return 1 - np.square(dist)


def get_best_sim(embeds, img_embed):
    most_sm = {"name": None, "sim": -np.inf}
    for key in embeds.keys():
        for value in embeds[key]:
            sim = cos_sim(value, img_embed)
            if sim > most_sm["sim"]:
                most_sm["name"] = key
                most_sm["sim"] = sim
    return most_sm


def evaluate(model: Model, ds, embeds):
    predictions = []
    labels = []
    for x, y in tqdm(ds):
        ret = model.predict(x.astype(np.float64), verbose=False)[0]

        most_sm = get_best_sim(embeds, ret)
        print(most_sm, y)
        predictions.append(most_sm["name"])
        labels.append(y)
    report = classification_report(labels, predictions)
    print(report)


class EmbedLoader:
    def __init__(self, bank_dir) -> None:
        self.__bank_dir = bank_dir
        self.__dir_lit = listdir(bank_dir)
        self.__len = len(self.__dir_lit)

    def __next__(self):
        return self.__load()

    def __iter__(self):
        return self.__load()

    def __load(self):
        for key in self.__dir_lit:
            cur_dir = path.join(self.__bank_dir, key)
            for img_name in listdir(cur_dir):
                im = imread(path.join(cur_dir, img_name))
                im = im[np.newaxis, :, :, :]
                yield im, key

    def __len__(self) -> int:
        return self.__len * 51


def main():
    config = None
    with open("./config.yaml", "r") as cf:
        config = yaml.load(cf, Loader=yaml.Loader)
        configdir = config["dir"]

    model = build_face_model(100)
    model.load_weights(config["weight"])
    model = Sequential(model.layers[:-1])

    model.summary()

    embeds = load_embeds(configdir["hashbank"])
    ds = EmbedLoader(configdir["facebankTest"])
    evaluate(model, ds, embeds)


if __name__ == "__main__":

    with tf.device("/cpu:0"):
        main()
