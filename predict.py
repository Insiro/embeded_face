from collections import defaultdict
from os import listdir, mkdir, path
from shutil import rmtree

import numpy as np
from numpy.linalg import norm
import tensorflow as tf
import yaml
from cv2 import imread
from keras import Model
from tqdm import tqdm
import json

from models.model import build_face_model


def gen_embeds(model: Model, regit_dir: str, embed_dir: str):
    """generate embed bank from regist imgs
    @param model: loaded Model
    @param regit_dir: path that containing regist imgs
    @param embed_dir: path form save embed results
    """
    if path.isdir(embed_dir):
        rmtree(embed_dir)
    mkdir(embed_dir)

    for user in tqdm(listdir(regit_dir)):
        embeds = []
        cur_folder = path.join(regit_dir, user)
        out_name = path.join(embed_dir, user) + ".fh"
        for img_name in listdir(cur_folder):
            img = imread(path.join(cur_folder, img_name))
            img = img[np.newaxis, :, :, :]
            pred = model(img)
            embeds.extend(pred.tolist())

        with open(path.join(out_name), "wb") as f:
            np.save(f, np.array(embeds))


def load_embeds(bank_dir: str):
    """load embed lists
    @param bank_dir : path for load face embeds"""
    embed_list = defaultdict(list)
    for hash_name in listdir(bank_dir):
        file_name = path.join(bank_dir, hash_name)
        if not path.isfile(file_name):
            continue
        label = ".".join(hash_name.split(".")[:-1])
        with open(file_name, "rb") as f:
            embed_list[label] = np.load(f)
    return embed_list


def cos_sim(x, y):
    a = np.dot(x, y)
    b = norm(x) * norm(y)
    sim = np.divide(a, b)
    return sim


def predict(model: Model, embeds, img):
    """
    @param model : The loaded model for predict image
    @param embeds : the list of registed image embeds
    @param img : image for predicted
    """
    img = img[np.newaxis, :, :, :]

    ret = model.predict(img)[0]

    most_sm = {"name": None, "sim": -np.Inf}
    sims = []

    for key in embeds.keys():
        for value in embeds[key]:
            sim = cos_sim(value, ret)
            if sim > most_sm["sim"]:
                most_sm["name"] = key
                most_sm["sim"] = sim
            sims.append({"id": key, "value": sim})
    sims.sort(key=lambda x: x["value"])
    print(json.dumps(sims, indent=2))
    return most_sm


def main():
    dir_config = None
    with open("./config.yaml", "r") as cf:
        dir_config = yaml.load(cf, Loader=yaml.Loader)["dir"]

    model = build_face_model(100)
    model.load_weights("./best_loss0.20312_acc0.890")

    # gen_embeds(model, dir_config["facebank"], dir_config["hashbank"])
    embeds = load_embeds(dir_config["hashbank"])

    img = imread("/data/01.datasets/FaceBank/Trained_Test/ID22/ID22-F-W0N1G_O.bmp")
    ids = predict(model, embeds, img)
    print(ids)


if __name__ == "__main__":

    with tf.device("/cpu:0"):
        main()
