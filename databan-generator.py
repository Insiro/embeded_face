from os import path, mkdir, listdir
from shutil import copy2, rmtree
from tqdm import tqdm
BASE_DIR = './MorphedImage'

paths = {
    'base' :BASE_DIR,
    'male' :path.join(BASE_DIR, 'Male'),
    'female' :path.join(BASE_DIR, 'Female'),
    'new' : './MorphedImageBank'
}


def delete_all():
    if path.isdir(paths['new']):
        rmtree(paths['new'])
    mkdir(paths['new'])

def gen_copy(target):
    target_dir = paths[target]
    for dir in tqdm( listdir(target_dir)):
        out_folder =path.join(paths['new'],target+'_'+dir)
        cur_folder =path.join(target_dir, dir)
        mkdir(out_folder)
        for img in listdir(cur_folder):
            cur = path.join(cur_folder, img)
            new = path.join(out_folder, img)
            copy2(cur, new)

def main():
    delete_all()
    gen_copy('male')
    gen_copy('female')
main()