import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from xgboost import XGBClassifier
#from root_numpy import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import uproot
#%jsroot on
#ROOT.gStyle.SetOptStat(0)

file_sig = uproot.open("WH_mA20_Mu_twoB.root:tree")
file_bkg = uproot.open("TTSL_2018_Mu_twoB.root:tree")

sigV1 = file_sig["evt_wgt"].array(library="np")
sigV2 = file_sig["leppt"].array(library="np")
sigV3 = file_sig["m_gg"].array(library="np")
sigV4 = file_sig["m_bbgg"].array(library="np")
sigV5 = file_sig["yout"].array(library="np")

bkgV1 = file_bkg["evt_wgt"].array(library="np")
bkgV2 = file_bkg["leppt"].array(library="np")
bkgV3 = file_bkg["m_gg"].array(library="np")
bkgV4 = file_bkg["m_bbgg"].array(library="np")
bkgV5 = file_bkg["yout"].array(library="np")

datasetSig_tuple = []
for i in range(len(sigV1)):
	temp = (sigV1[i], sigV2[i], sigV3[i], sigV4[i], sigV5[i])
	datasetSig_tuple.append(temp)

ntuple_datasetSig = np.array(datasetSig_tuple)

datasetBkg_tuple = []
for i in range(len(bkgV1)):
        temp = (bkgV1[i], bkgV2[i], bkgV3[i], bkgV4[i], bkgV5[i])
        datasetBkg_tuple.append(temp)

ntuple_datasetBkg = np.array(datasetBkg_tuple)
print(ntuple_datasetBkg.shape)


XSig = ntuple_datasetSig[:,1:4]   ### pick elements from 0 to 3
YSig = ntuple_datasetSig[:,4]
XBkg = ntuple_datasetBkg[:,1:4]
YBkg = ntuple_datasetBkg[:,4]
#print(XSig)
print(YSig)
#print(XBkg)
print(YBkg)

seed = 7
test_size = 0.33
X_trainSig, X_testSig, Y_trainSig, Y_testSig = train_test_split(XSig, YSig, test_size=test_size, random_state=seed)
X_trainBkg, X_testBkg, Y_trainBkg, Y_testBkg = train_test_split(XBkg, YBkg, test_size=test_size, random_state=seed)
#print(X_trainSig)
#print(X_trainBkg)
#print(Y_trainSig)
#print(Y_trainBkg)
X_train = np.vstack((X_trainSig, X_trainBkg))
X_test = np.vstack((X_testSig,X_testBkg))
Y_train = np.hstack((Y_trainSig,Y_trainBkg))
Y_test = np.hstack((Y_testSig,Y_testBkg))
#print(X_train)
#print(X_test)
#print(Y_train)
#print(Y_test)

# fit model no training data
model = XGBClassifier()
gbm = model.fit(X_train, Y_train)
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

############################################ ROC Curve

# Compute micro-average ROC curve and ROC area
fpr, tpr, _ = roc_curve(Y_test, y_pred)
roc_auc = auc(fpr, tpr)
gbm.get_booster().feature_names = ["pt_lep", "m_gg", "m_bbgg"]
xgb.plot_importance(gbm)
#plt.show()
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
##################################################

roc_auc1 = auc(fpr, tpr)                                                                                                                                                                                  
plt.plot(fpr, tpr, lw=1, label='Test ROC (area = %0.2f)'%(roc_auc1))                                                                                                                                      
fpr2, tpr2, thresholds2 = roc_curve(Y_test, y_pred)                                                                                                                                            
roc_auc2 = auc(fpr2, tpr2)                                                                                                                                                                                  
plt.plot(fpr2, tpr2, lw=1, label='Train ROC (area = %0.2f)'%(roc_auc2))                                                                                                                                     
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')                                                                                                                                         
plt.xlim([-0.05, 1.05])                                                                                                                                                                                     
plt.ylim([-0.05, 1.05])                                                                                                                                                                                     
plt.xlabel('False Positive Rate')                                                                                                                                                                           
plt.ylabel('True Positive Rate')                                                                                                                                                                            
plt.title('Receiver operating characteristic')                                                                                                                                                              
plt.legend(loc="lower right")                                                                                                                                                                               
plt.grid()                                                                                                                                                                                                  
plt.show()
