import pandas as pd
from PIL import Image
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import os
import sys

ROOT_DIR = os.getcwd()

if len(sys.argv[1:]) == 0:
    exit()
if len(sys.argv[1:]) == 1:
    csv, path = sys.argv[1], ROOT_DIR+"/"
if len(sys.argv[1:]) == 2:
    csv, path = sys.argv[1], sys.argv[2]




data = pd.read_csv(ROOT_DIR+"/"+csv)

def processFile(x):
    img = Image.open("/home/stagiaire/SSD/yannis/images_v3_seq20/"+str(x))
    img = np.array(img.resize((224,224)))
    return (img, int(x[0]))


dt = h5py.special_dtype(vlen=str)

nameh5 = csv.replace(".csv", ".h5").replace("_labels", "")

with h5py.File(path+"/"+nameh5, 'w') as hff:
    img_ds = hff.create_dataset("images", shape=(1, 224, 224, 3), maxshape=(None, 224, 224, 3), dtype=np.uint8)
    labels_ds = hff.create_dataset("labels", shape=(1, ), maxshape=(None, ), dtype='i1')
    
    results = Parallel(n_jobs=30)(delayed(processFile)(x) for x in tqdm(data["name"].tolist())  )
    first = True
    for x in tqdm(results):
        img = x[0]
        lb = x[1]
        if first:
            img_ds[0:] = img
            labels_ds[0] = lb
            first = False
        else:
            img_ds.resize((img_ds.shape[0] + 1), axis=0)
            img_ds[-1:] = img
            labels_ds.resize((labels_ds.shape[0]+1), axis=0)
            labels_ds[-1] = lb
