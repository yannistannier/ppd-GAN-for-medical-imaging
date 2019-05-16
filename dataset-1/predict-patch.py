import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pandas as pd
from keras.models import load_model
import glob
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def patch3(img):
    width, height = img.size
    new_width = height
    new_height = height
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    name_p = "patch3"

    img1 = img.crop((0, 0, height, height))
    img2 = img.crop((width-height, 0, width, height))
    img3 = img.crop((left, top, right, bottom))
    
    return (img1, img2, img3)


def patch5(image):
    width, height = image.size
    square = 230
    left = square + (square/2)
    top = 0 
    right = left + square
    bottom = top + square
    img1 = image.crop((left, top, right, bottom))
    
    square = 230
    left = 0 + (square/2)
    top = 0 
    right = left + square
    bottom = top + square
    img2 = image.crop((left, top, right, bottom))
    
    square = 230
    left = 0 + (square/2)
    top = 230 
    right = left + square
    bottom = top + square
    img3 = image.crop((left, top, right, bottom))
    
    square = 230
    left = square + (square/2)
    top = 230 
    right = left + square
    bottom = top + square
    img4 = image.crop((left, top, right, bottom))
    
    square = 230
    left = (width - square)/2
    top = (height - square)/2
    right = (width + square)/2
    bottom = (height + square)/2

    img5 = image.crop((left, top, right, bottom))
    
    return (img1, img2, img3, img4, img5)


test = pd.read_csv("test.csv")

x_test_96 = []
x_test_224 = []
y_test = []
y_test_p = []

for img in tqdm(test.iterrows()):
    i = img[1]
    image_orig = Image.open("data/orig_test/"+i["name"])

    patchs = patch5(image_orig)
    
    for mi in patchs:
        y_test.append(i["label"])
        m = mi.resize((224,224))
        x_test_224.append(np.array(m)*(1/255))

        m = mi.resize((96,96))
        x_test_96.append(np.array(m)*(1/255))

x_test_96 = np.array(x_test_96)
x_test_224 = np.array(x_test_224)
y_test = np.array(y_test)


all_models = glob.glob("../models/dataset-1/patch5/*/*")

res = {}
# res["y_true"] = y_test
res["y_test"] = y_test

for m in all_models:
    model = load_model(m, compile=False)
    if "224x224" in m:
        pred = model.predict(x_test_224, verbose=1)
    if "96x96" in m:
        pred = model.predict(x_test_96, verbose=1)
    res[m.split("/")[-2]] = pred[:,1]

res = pd.DataFrame(res)
res.to_csv("resultat_patch3_orig.csv", index=False)