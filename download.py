import os
import glob
import zipfile
import shutil
import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------ dataset-1 ---------
print("Download Dataset 1 ....")
d1 = "dataset-1/data"
os.makedirs(d1, exist_ok=True)
os.system("wget https://s3-eu-west-1.amazonaws.com/ppd-gan-medical-imaging/breast-cancer-1.zip -P "+d1)
print("Unzip ...")
zip_ref = zipfile.ZipFile(d1+"/breast-cancer-1.zip", 'r')
zip_ref.extractall(d1)
zip_ref.close()
os.remove(d1+"/breast-cancer-1.zip")


# # ------ dataset-2 ---------
# print("Download Dataset 2 ...")
# d2 = "dataset-2/data"
# os.makedirs(d2, exist_ok=True)
# os.system("wget https://s3-eu-west-1.amazonaws.com/ppd-gan-medical-imaging/pneumonia-dataset-2.zip -P "+d2)
# zip_ref = zipfile.ZipFile(d2+"/pneumonia-dataset-2.zip", 'r')
# zip_ref.extractall(d2)
# zip_ref.close()

# fileszip = glob.glob(d2+"/t*.zip")
# for z in fileszip:
#     zip_ref = zipfile.ZipFile(z, 'r')
#     zip_ref.extractall(d2)
#     zip_ref.close()
#     os.remove(z)

    
# # ------ dataset-3 ---------
# print("Download Dataset 2 ....")
# d3 = "dataset-3/data"
# os.makedirs(d3, exist_ok=True)
# os.system("wget https://s3-eu-west-1.amazonaws.com/ppd-gan-medical-imaging/histopathologic-cancer-detection.zip -P "+d3)
# print("Unzip ....")
# zip_ref = zipfile.ZipFile(d3+"/histopathologic-cancer-detection.zip", 'r')
# zip_ref.extractall(d3)
# zip_ref.close()

# fileszip = glob.glob(d3+"/t*.zip")
# for z in fileszip:
#     dir_dest = d3+"/orig_"+z.split('/')[-1][:-4]
#     os.makedirs(dir_dest,exist_ok=True)
#     zip_ref = zipfile.ZipFile(z, 'r')
#     zip_ref.extractall(dir_dest)
#     zip_ref.close()
#     os.remove(z)


# df = pd.read_csv(d3+"/train_labels.csv")
# train, test = train_test_split(df, stratify = df["label"], random_state=1, test_size=0.08)

# for x in tqdm(train.iterrows()):
#     nm = str(x[1]["id"])
#     os.makedirs(d3+"/train", exist_ok=True)
#     shutil.copy(d3+"/orig_train/"+nm+".tif", d3+"/train/"+nm+".tif")

# for x in tqdm(train.iterrows()):
#     nm = str(x[1]["id"])
#     os.makedirs(d3+"/val", exist_ok=True)
#     shutil.copy(d3+"/orig_train/"+nm+".tif", d3+"/val/"+nm+".tif")
