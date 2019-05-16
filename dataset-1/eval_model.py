from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, recall_score, precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve



def plot_confusion_matrix(cm, classes=["legit","faude"],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    #plt.tight_layout()

def calcul_curve(prediction16, y_true, max_frr = 0.029):
    list_FAR = []
    list_FRR = []
    t = np.arange(0.0, 1.01, 0.02)
    EER = 1
    best_frr = [0.0,1.0,0.0]
    for x in t:
        y_pred = (prediction16[:,0] < x).astype(np.int)
        cm = confusion_matrix(y_true, y_pred).T
        TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        FRR = FN / float(FN + TP)
        FAR = FP / float(TN + FP)
        list_FAR.append(FAR)
        list_FRR.append(FRR)
        EER = x if FAR - FRR < EER and FAR - FRR > 0 else EER

    for x, v in enumerate(list_FRR):
        if v < max_frr and v > 0:
            best_frr = [t[x], v, list_FAR[x]]

    print("--------------------------------------")
    print("best result for max FRR ", max_frr,", threeshold = ", best_frr[0])
    print(" FRR : ", best_frr[1])
    print(" FAR : ", best_frr[2])
    print("--------------------------------------")
    
    plt.figure(figsize=(13, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, list_FAR, label="FAR") # plotting t, a separately 
    plt.plot(t, list_FRR,  label="FRR") # plotting t, b separately
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('%')
    plt.xlabel('threshold (EER = '+str(EER)+')')
    
    plt.subplot(1, 2, 2)
    calcul_score(prediction16, y_true, best_frr[0], False)
    
    plt.tight_layout()	


def calcul_score(prediction, y_true, threshold, text=True):
    
    def compute_FRR(TP, FP, FN, TN):
        FRR = FN / float(FN + TP)
        return FRR

    def compute_FAR(TP, FP, FN, TN):
        FAR = FP / float(TN + FP)
        return FAR

    y_pred = (prediction[:,0] < threshold).astype(np.int)
    cm = confusion_matrix(y_true, y_pred).T
    acc = accuracy_score(y_true, y_pred)
    plot_confusion_matrix(cm)
    if text :
        print("====== Accuracy : ", acc)
        TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        print(" ==== FRR :", compute_FRR(TP, FP, FN, TN))
        print(" ==== FAR :", compute_FAR(TP, FP, FN, TN))
    else:
        print("accuracy : ", acc)


# def calcul_curve(prediction16, y_true):
#     list_FAR = []
#     list_FRR = []
#     t = np.arange(0.0, 1.01, 0.05)
#     EER = 1
#     for x in t:
#         y_pred = (prediction16[:,0] < x).astype(np.int)
#         cm = confusion_matrix(y_true, y_pred).T
#         TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
#         FRR = FN / float(FN + TP)
#         FAR = FP / float(TN + FP)
#         list_FAR.append(FAR)
#         list_FRR.append(FRR)
#         EER = x if FAR - FRR < EER and FAR - FRR > 0 else EER
#     plt.plot(t, list_FAR, label="FAR") # plotting t, a separately 
#     plt.plot(t, list_FRR,  label="FRR") # plotting t, b separately
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.ylabel('%')
#     plt.xlabel('threshold (EER = '+str(EER)+')')
#     plt.show()

def calcul_softmax(threshold = 0.5):
    y_pred = 1-(prediction[:,0] < threshold).astype(np.int)
    return y_pred

def calcul_sigmoid(threshold = 0.5):
    y_pred = np.where(prediction > threshold, 1, 0)
    return y_pred




def show_error(prediction, y_true, images, type_error, threshold=0.5):
    def plot_figures(figures, nrows = 1, ncols=1):
        """Plot a dictionary of figures.

        Parameters
        ----------
        figures : <title, figure> dictionary
        ncols : number of columns of subplots wanted in the display
        nrows : number of rows of subplots wanted in the figure
        """

        fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(25,25)
        for ind,title in enumerate(figures):
            axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
            axeslist.ravel()[ind].set_title(title)
            axeslist.ravel()[ind].set_axis_off()
        plt.tight_layout() # optionalplt.imshow(x[10])

    
    
    y_pred = (prediction[:,0] < threshold).astype(np.int)
    errors = np.where( (y_pred == y_true[0:]) == False )[0]
    
    img_error = []
    
    for index in errors:
        if type_error == "FN" and y_true[index] == 1:
            img_error.append([index,images[index]])
        if type_error == "FP" and y_true[index] == 0:
            img_error.append([index, images[index]])
        if len(img_error) == 100:
            break
    
    # generation of a dictionary of (title, images)
    number_of_im = len(img_error)
    figures = {type_error+' '+str(img[0]): img[1] for img in img_error}

    # plot of the images in a figure, with 2 rows and 3 columns
    plot_figures(figures, 10, 10)





def compare_model(list_predictions, name, y_true):
    models_FAR = []
    models_FRR = []
    t = np.arange(0.0, 1.01, 0.5)
    
    for preds in list_predictions:
        list_FAR=[]
        list_FRR=[]
        for x in t:
            y_pred = (preds[:,0] < x).astype(np.int)
            cm = confusion_matrix(y_true, y_pred).T
            TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            FRR = FN / float(FN + TP)
            FAR = FP / float(TN + FP)
            list_FAR.append(FAR)
            list_FRR.append(FRR)
        models_FAR.append(list_FAR)
        models_FRR.append(list_FRR)
    
    
    for i, far in enumerate(models_FAR):
        plt.plot(t, far, label="FAR - "+name[i])
    plt.legend(loc=9)
    plt.show()
    
    for i, frr in enumerate(models_FRR):
        plt.plot(t, frr, label="FRR - "+name[i])
        
    plt.legend(loc=9)
    plt.show()
    
# from keras.models import load_model
# from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, recall_score, precision_score
# from keras import backend as K
# from keras.preprocessing.image import ImageDataGenerator
# import pandas as pd
# import numpy as np
# import h5py


# class EvalModel:
#     def __init__(self, path_model, size=(224, 224, 3) ):
#         print("loading model ...")
#         self.Model = load_model(path_model)
#         self.batch_size_test = 1
#         self.steps = 100
#         #self.TP, self.FP, self.FN, self.TN = None,None,None,None
#         self.test_generator = None
#         self.test_set = None
#         self.size = size
    
#     def set_test_generator(self, path_csv, path_test):
#         test_datagen = ImageDataGenerator(rescale=1./255)
#         self.path_csv=path_csv
#         self.test_generator = test_datagen.flow_from_dataframe(
#             pd.read_csv(path_csv),
#             path_test,
#             x_col="name",
#             y_col="label",
#             has_ext=True,
#             target_size=self.size[:-1],
#             batch_size=self.batch_size_test,
#             class_mode="categorical",
#             shuffle=True
#         )
    
    
#     def set_test_h5(self, path):
#         self.test_set = h5py.File(path,'r')
    
#     def run_test(self):
        
#         images = self.test_set["images"]
        
#         y_pred = self.Model.predict(images, verbose=1)

#         y_true = self.test_set["labels"]
        
#         acc, FPR, FAR = self.calcul_scores(y_true, y_pred)
#         print("acc : ", acc)
#         print("FPR : ", FPR)
#         print("FAR : ", FAR)
    
    
#     def calcul_scores(self, y_true, y_pred, threshold=0.1):
#         print(y_pred.shape[1])
        
#         if y_pred.shape[1] == 2 :
#             y_pred = 1-(y_pred[:,0] < threshold).astype(np.int)
        
#         if y_pred.shape[1] == 1 :
#             y_pred = 1-np.where(y_pred > threshold, 0, 1)
            
# #             y_pred = 1 - np.argmax(y_pred, axis=1)
# #         y_pred = K.cast(y_pred >= threshold, 'float32')
# #         y_pred = (y_pred > threshold).astype(int)
# #         print(y_pred)
# #         y_pred = np.where(y_pred > threshold, 1, 0)

# #         y_pred = (y_pred[:,0] < threshold).astype(np.int)
#         #
# #         print(y_pred)
        
#         cm = confusion_matrix(y_true, y_pred)
#         print(cm)
#         acc = accuracy_score(y_true, y_pred)
#         TP, FP, FN, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
#         FPR = self.compute_FRR(TP, FP, FN, TN)
#         FAR = self.compute_FAR(TP, FP, FN, TN)
#         return acc, FPR, FAR
    
    
#     # false recognition rate (FRR) : FRR = FNR = FN/(FN + TP)
#     def compute_FRR(self, TP, FP, FN, TN):
#         FRR = FN / float(FN + TP)
#         return FRR

#     # false acceptance rate (FAR) : FAR = FPR = FP/(FP + TN)
#     def compute_FAR(self, TP, FP, FN, TN):
#         FAR = FP / float(TN + FP)
#         return FAR

