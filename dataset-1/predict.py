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

test = pd.read_csv("test.csv")

x_test_96 = []
x_test_224 = []
y_test = []

for img in tqdm(test.iterrows()):
    i = img[1]
    y_test.append(i["label"])
    mi = Image.open("data/orig_test/"+i["name"])
    
    m = mi.resize((224,224))
    x_test_224.append(np.array(m)*(1/255))

    m = mi.resize((96,96))
    x_test_96.append(np.array(m)*(1/255))

x_test_96 = np.array(x_test_96)
x_test_224 = np.array(x_test_224)
y_test = np.array(y_test)


all_models = glob.glob("../models/dataset-1/patch3/*/*")

res = {}
res["y_true"] = y_test

for m in all_models:
    model = load_model(m, compile=False)
    if "224x224" in m:
        pred = model.predict(x_test_224, verbose=1)
    if "96x96" in m:
        pred = model.predict(x_test_96, verbose=1)
    res[m.split("/")[-2]] = pred[:,1]

res = pd.DataFrame(res)
res.to_csv("resultat_patch3_orig.csv", index=False)