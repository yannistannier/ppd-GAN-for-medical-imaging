## PPD : GANs for medical imaging

## Dataset 1 : Breast Cancer 
Kaggle & evaluation : https://www.kaggle.com/t/77a71e6879604ca9baac71fb01251139  
dataset : https://drive.google.com/drive/folders/1JaXZuUoK3L95mD2ySMjaTG2jtUw1_A7A?usp=sharing

Total image : 7909 images  
Training size : 6327 ( 1 => 4343, 0 => 1984 )    
Test size : 1582

Result :  

1. Baseline & Patch:  

|                           |               |            |         |
| :------------------------ | :------------ | :--------- | :------ |
| Model                    | Original data | Patch 3    | Patch 5 |
| *CNN\_224x224*            | 91.94%        | **92.82%** | 90.94%  |
| *CNN\_128x128*            | 89.15%        | 90.45%     | 89.71%  |
| *CNN\_224x224\_StainNorm* | 88.95%        | 89.95%     | 89.35%  |
| *CNN\_128x128\_StainNorm* | 85.63%        | 86.04%     | 87.91%  |



2. Comparaison gan's data training

| **Number of images used in training** | **7k** | **12k** | **20k** | **30k**    |
| :---------------------------------- | :----- | :------ | :------ | :--------- |
| Real Data - baseline                | 92.82% | \-      | \-      | \-         |
| ProgressiveGan data                 | 90.06% | 91.54%  | 90.91%  | 92.86%     |
| StyleGan data                       | 83.15% | 81.00%  | 81.45%  | 84.54%     |
| Mixed data                          | 94.32% | 93.9%   | 94.51%  | **94.63%** |


2. Comparaison real data + gan's data

| **Augmented Training Data**     | **Accuracy** |
| :---------------------------------------- | :----------- |
| CNN - Real Data                           | 92.82%       |
| CNN - Real Data + Progressive Growing GAN | 95.45%       |
| CNN - Real Data + Style GAN               | 95.75%       |
| CNN - Real Data + Mixed data from GANs    | **96.15%**   |

## Dataset 2 : Pneumonia Detection
Kaggle & evaluation : https://www.kaggle.com/t/713c49b3cf224a80b4e082936e31bd17   
dataset : https://drive.google.com/drive/folders/1Dm3yMFz023iQukgZ3h4dQ8dDNEpcEwbO?usp=sharing

Total image : 26684 images  
Training size : 24015 ( 0 => 18604, 1 => 5411 )    
Test size : 2669


Result :


|                   **Model**                    | **Accuracy** |
| :--------------------------------------------: | :----------: |
|              CNN - Real Data 26k               |    88.15%    |
|     CNN - Progressive Growing Gan Data 26k     |    84.24%    |
| CNN - Real Data + Progressive Growing Gan Data |  **92.02%**  |



## Dataset 3 : Skin Cancer
Kaggle : https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

Total image : 10000

Result :


|                      | 2 class | 2 class | 2 class | 7 class |
|----------------------|:-------:|---------|---------|---------|
|         Model        |   Acc   | ROC auc |  PR auc |   Acc   |
| CNN_224x224 weighted |  91.11% | 93.78%  | 82.06%  | 88.36%  |
| CNN_224x224          |  90.41% | 93.78%  | 81.77%  | 87.86%  |
| CNN - Gan            | 91.51%  | 93.97%  | 82.71%  | 90.36%  |

