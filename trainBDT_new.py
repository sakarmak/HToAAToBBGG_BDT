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
import seaborn
#%jsroot on
#ROOT.gStyle.SetOptStat(0)

### Get the root file with the tree name
file_sig = uproot.open("WH_m55_Mu_twoB.root:tree")
file_bkg = uproot.open("TTbar_Mu_twoB.root:tree")

### Store the branches as pandas dataframe
#sig = file_sig.arrays(["evt_wgt", "leppt","m_gg", "m_bb", "m_bbgg", "yout"], library="pd")
#bkg = file_bkg.arrays(["evt_wgt", "leppt","m_gg", "m_bb", "m_bbgg", "yout"], library="pd")
sig = file_sig.arrays(["evt_wgt", "leppt", "m_bb", "m_bbgg", "met", "n_jet", "yout"], library="pd")
bkg = file_bkg.arrays(["evt_wgt", "leppt", "m_bb", "m_bbgg", "met", "n_jet", "yout"], library="pd")

#sig_corr = file_sig.arrays([ "leppt", "m_gg", "m_bb", "m_bbgg"], library="pd")
#bkg_corr = file_sig.arrays([ "leppt", "m_gg", "m_bb", "m_bbgg"], library="pd")
sig_corr = file_sig.arrays(["leppt", "m_bb", "m_bbgg", "met", "n_jet"], library="pd")
bkg_corr = file_sig.arrays(["leppt", "m_bb", "m_bbgg", "met", "n_jet"], library="pd")
#print(sig)
#print(bkg)

### name of the variables used in training, will be used in feature importance plot
name_Var = ["leppt", "m_bb", "m_bbgg", "met", "n_jet"]

### Store the data as numpy array from the pandas dataframe
ntuple_datasetSig = sig.to_numpy()
ntuple_datasetBkg = bkg.to_numpy()
#print(ntuple_datasetSig.shape)
#print(ntuple_datasetBkg.shape)


XSig = ntuple_datasetSig[:,1:6]   ### pick elements from 0 to 5 
YSig = ntuple_datasetSig[:,6]
XBkg = ntuple_datasetBkg[:,1:6]
YBkg = ntuple_datasetBkg[:,6]
wgt_sig = ntuple_datasetSig[:,0]
wgt_bkg = ntuple_datasetBkg[:,0]
#print(XSig)
#print(YSig)
#print(XBkg)
#print(YBkg)

seed = 7
test_size = 0.33
X_trainSig, X_testSig, Y_trainSig, Y_testSig, wgt_trainSig, wgt_testSig = train_test_split(XSig, YSig, wgt_sig, test_size=test_size, random_state=seed)
X_trainBkg, X_testBkg, Y_trainBkg, Y_testBkg, wgt_trainBkg, wgt_testBkg = train_test_split(XBkg, YBkg, wgt_bkg, test_size=test_size, random_state=seed)
print("X_trainSig shape: ",X_trainSig.shape)
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
model = XGBClassifier(learning_rate=0.3,max_depth=6)
gbm = model.fit(X_train, Y_train)
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

############################################ ROC Curve

# Compute micro-average ROC curve and ROC area
fpr, tpr, _ = roc_curve(Y_test, y_pred)
roc_auc = auc(fpr, tpr)
gbm.get_booster().feature_names = name_Var
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
# Plot correlation matrix for signal and backgrounds
correlation_sig = sig_corr.corr()
seaborn.heatmap(correlation_sig).set(title = "sig correlation")
plt.show()

correlation_bkg = bkg_corr.corr()
seaborn.heatmap(correlation_bkg).set(title = "bkg correlation")
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
#################################################
# Plot the output BDT distribution of Signal and Background
y_pred_proba_real = y_pred_proba[:,1]
y_pred_sig = []
y_pred_bkg = []
for i in range (len(Y_test)):
    if Y_test[i]==1:
        y_pred_sig.append(y_pred_proba_real[i])
    else:
        y_pred_bkg.append(y_pred_proba_real[i])
plt.hist([y_pred_sig, y_pred_bkg],color=['g','r'],histtype='step',bins=40, label=["Signal", "Background"])
#plt.hist([y_pred_sig, y_pred_bkg],color=['g','r'],histtype='step',bins=40, label=["Signal", "Background"], density=[True, True])
#plt.yscale("log")
plt.legend()
plt.show()
##################################################
# Plot the different BDT input variables
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = [7.50, 7.50]
plt.rcParams["figure.autolayout"] = True
"""Nbins=[100,200,90,50,100,20]
Range=[[0,200],[10,70],[0,90],[50,200],[0,500],[0,20]]
Ncols = 2
Nrows = 3
xAxisLabel = ["$p_{T}^{l}$", "$m_{\gamma\gamma}$", "$m_{bb}$", "$m_{bb\gamma\gamma}$", "Met", "$n_{jet}$"]"""
Nbins=[100,90,50,100,20]
Range=[[0,200],[0,90],[50,200],[0,500],[0,20]]
Ncols = 2
Nrows = 3
xAxisLabel = ["$p_{T}^{l}$", "$m_{bb}$", "$m_{bb\gamma\gamma}$", "$p_{T}^{miss}$", "$n_{jet}$"]
figure, axis = plt.subplots(Nrows, Ncols)
counter=0
for i in range(Nrows):
    for j in range(Ncols):
        if (i+j+counter< len(Nbins)):
            axis[i,j].hist([X_testSig[:,i+j+counter], X_testBkg[:,i+j+counter]], color=['g','r'], histtype='stepfilled', bins=Nbins[i+j+counter], label=["Signal", "Background"], density=[True,True], range=Range[i+j+counter], alpha = 0.5)
            axis[i,j].set_xlabel(xAxisLabel[i+j+counter], fontsize=20)
            axis[i,j].set_ylabel("AU")
            axis[i,j].legend()
    counter = counter+Ncols-1
#plt.hist([X_testSig[:,2], X_testBkg[:,2],],color=['g','r'],histtype='step',bins=60, label=["Signal", "Background"], density=[True,True], range=[0,60])
plt.subplots_adjust(left=0.1, bottom=0.12, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
