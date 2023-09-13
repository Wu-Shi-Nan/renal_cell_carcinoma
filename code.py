#%%导入包
from sklearn.ensemble import VotingClassifier
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import miceforest as mf
from sklearn.datasets import load_iris
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn import metrics
from matplotlib import pyplot
from numpy import argmax
from functools import reduce
from sklearn import metrics
import seaborn as sns
from sklearn import tree
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import streamlit as st
import pickle
import sklearn
import json
import random
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import shap
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score
# %matplotlib.inline
%matplotlib
sns.set()
#%%随机数种子和交叉验证的次数0
random_state_new = 50
jc = 10
#%% 导入相关的数据
train_data = pd.read_csv('/Users/mac/Desktop/吴世楠文件夹/肾癌翻修/renal_cancer_ml (2)/train.csv')
test_data = pd.read_csv('/Users/mac/Desktop/吴世楠文件夹/肾癌翻修/renal_cancer_ml (2)/test.csv')
#%%自变量和因变量的选择
features = ["Sequence.number","Primary.Site","Grade","T",'N','surgery','Radiation','Chemotherapy','Systemic','Tumor.Size']
indicator = ["M"]
X_train = train_data[features] 
X_test = test_data[features]
y_train = train_data[indicator]
y_test = test_data[indicator]
y_train = label_binarize(y_train, classes=[0,1])
y_test = label_binarize(y_test, classes=[0,1])
X_train = np.array(X_train)
X_test = np.array(X_test)
X = X_train
y = y_train
#%%数据不平衡的处理
oversample = SMOTE(random_state=random_state_new)

X_train, y_train = oversample.fit_resample(X_train, y_train)

#%%建立模型
jc = 10
lr = LogisticRegression(penalty="none", random_state=random_state_new)
rf = RandomForestClassifier(
    n_estimators=200,  max_features=4, random_state=random_state_new)
dt = tree.DecisionTreeClassifier(
    min_weight_fraction_leaf=0.25, random_state=random_state_new)
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.01,
                    power_t=0.5,
                    max_iter=200,
                    shuffle=True, random_state=random_state_new)
gbm = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1, max_depth=1, random_state=random_state_new)
# =============================================================================
# 划分数据集
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state_new)
# =============================================================================
# 分类器模型的导入
# =============================================================================
# =============================================================================
# 逻辑回归模型
# =============================================================================
logis_model = LogisticRegression(random_state=random_state_new,
                                 solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
logis_model.score(X_test, y_test)
# =============================================================================
# K近邻分类模型
# =============================================================================
KNN_model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
KNN_model.score(X_test, y_test)
# =============================================================================
# 高斯贝叶斯分类模型
# =============================================================================
GaNB_model = GaussianNB().fit(X_train, y_train)
GaNB_model.score(X_test, y_test)
# =============================================================================
# 决策树分类模型
# =============================================================================
tree = tree.DecisionTreeClassifier(random_state=random_state_new)
tree_model = tree.fit(X_train, y_train)
tree_model.score(X_test, y_test)
# =============================================================================
# Bagging分类模型
# =============================================================================
Bag = BaggingClassifier(KNeighborsClassifier(
), max_samples=0.5, max_features=0.5, random_state=random_state_new)
Bag_model = Bag.fit(X_train, y_train)
Bag_model.score(X_test, y_test)
# =============================================================================
# 随机森林模型
# =============================================================================
RF = RandomForestClassifier(n_estimators=10, max_depth=3,
                            min_samples_split=12, random_state=random_state_new)
RF_model = RF.fit(X_train, y_train)
RF_model.score(X_test, y_test)
# =============================================================================
# 极端随机树分类模型
# =============================================================================
ET = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                          min_samples_split=2, random_state=random_state_new)
ET_model = ET.fit(X_train, y_train)
ET_model.score(X_test, y_test)
# =============================================================================
# AdaBoost模型
# =============================================================================
AB = AdaBoostClassifier(n_estimators=10, random_state=random_state_new)
AB_model = AB.fit(X_train, y_train)
AB_model.score(X_test, y_test)
# =============================================================================
# GBDT模型
# =============================================================================
GBT = GradientBoostingClassifier(
    n_estimators=10, learning_rate=1.0, max_depth=1, random_state=random_state_new)
GBT_model = GBT.fit(X_train, y_train)
GBT_model.score(X_test, y_test)
# =============================================================================
# VOTE模型
# =============================================================================
clf1 = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=random_state_new)
clf3 = GaussianNB()
VOTE = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
VOTE_model = VOTE.fit(X_train, y_train)
VOTE_model.score(X_test, y_test)
# =============================================================================
# GBM模型
# =============================================================================
gbm = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.01, max_depth=1, random_state=random_state_new)
gbm_model = gbm.fit(X_train, y_train)
gbm_model.score(X_test, y_test)
# =============================================================================
# XGboost模型
# =============================================================================
xgb_model = xgb.XGBClassifier(
    n_estimators=360, max_depth=1, learning_rate=0.7, random_state=random_state_new)
xgb_model = xgb_model.fit(X_train, y_train)
xgb_model.score(X_test, y_test)
# =============================================================================
# MLP模型
# =============================================================================
mlp_model = mlp.fit(X_train, y_train)
mlp_model.score(X_test, y_test)
lr_model = logis_model
#%%模型准确率的比较
print("Logistic回归的模型准确率：{:.3f}".format(logis_model.score(X_test, y_test)))
print("KNN回归的模型准确率：{:.3f}".format(KNN_model.score(X_test, y_test)))
print("高斯贝叶分类器的模型准确率：{:.3f}".format(GaNB_model.score(X_test, y_test)))
print("决策树分类器的模型准确率：{:.3f}".format(tree_model.score(X_test, y_test)))
print("Bagging分类模型的模型准确率：{:.3f}".format(Bag_model.score(X_test, y_test)))
print("随机森林分类模型的模型准确率：{:.3f}".format(RF_model.score(X_test, y_test)))
print("极端随机树分类模型的模型准确率：{:.3f}".format(ET_model.score(X_test, y_test)))
print("AdaBoost模型的模型准确率：{:.3f}".format(AB_model.score(X_test, y_test)))
print("GBDT模型的模型准确率：{:.3f}".format(GBT_model.score(X_test, y_test)))
print("VOTE模型的模型准确率：{:.3f}".format(VOTE_model.score(X_test, y_test)))
print("GBM模型的模型准确率：{:.3f}".format(gbm_model.score(X_test, y_test)))
print("XGboost模型的模型准确率：{:.3f}".format(xgb_model.score(X_test, y_test)))

#%%交叉验证
from sklearn.model_selection import cross_val_score,StratifiedKFold,LeaveOneOut
strKFold = StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
cv=strKFold
result_lr=cross_val_score(lr_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
result_ab=cross_val_score(AB_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
result_bag=cross_val_score(Bag_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
result_mlp=cross_val_score(mlp_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
result_gbm=cross_val_score(gbm_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
result_xgb=cross_val_score(xgb_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
#%%折线图的绘制
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
plt.plot(x, result_lr, label='LR: Average AUC = {:.3f}, STD = {:.3f}'.format(result_lr.mean(), result_lr.std()),
         linewidth=3, color='#fe5722', marker='>', markerfacecolor='#fe5722', markersize=12)
plt.plot(x, result_ab, label='AB: Average AUC = {:.3f}, STD = {:.3f}'.format(result_ab.mean(), result_ab.std()),
         linewidth=3, color='#03a8f3', marker='>', markerfacecolor='#03a8f3', markersize=12)
plt.plot(x, result_mlp, label='MLP: Average AUC = {:.3f}, STD = {:.3f}'.format(result_mlp.mean(), result_mlp.std()),
         linewidth=3, color='#009587', marker='>', markerfacecolor='#009587', markersize=12)
plt.plot(x, result_bag, label='BAG: Average AUC = {:.3f}, STD = {:.3f}'.format(result_bag.mean(), result_bag.std()),
         linewidth=3, color='#673ab6', marker='>', markerfacecolor='#673ab6', markersize=12)
plt.plot(x, result_gbm, label='GBM: Average AUC = {:.3f}, STD = {:.3f}'.format(result_gbm.mean(), result_gbm.std()),
         linewidth=3, color='#b5da3d', marker='>', markerfacecolor='#b5da3d', markersize=12)
plt.plot(x, result_xgb, label='XGB: Average AUC = {:.3f}, STD = {:.3f}'.format(result_xgb.mean(), result_xgb.std()),
         linewidth=3, color='#3f51b4', marker='>', markerfacecolor='#3f51b4', markersize=12)
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.ylim(0.5, 1.00)
plt.xlim(0.7, 10.3)
plt.xlabel('Round of Cross')
plt.ylabel('AUC')
plt.title('Ten Fold Cross Validation')
plt.legend(loc=4)
plt.show()
#%%train and test ROC曲线
plt.style.use('tableau-colorblind10')
def plot_roc(k,y_pred_undersample_score,labels_test,classifiers,color,title):
    fpr, tpr, thresholds = metrics.roc_curve(labels_test.values.ravel(),y_pred_undersample_score)
    roc_auc = metrics.auc(fpr,tpr)
    plt.figure(figsize=(20,16))
    plt.figure(k)
    plt.title(title)
    plt.plot(fpr, tpr, 'b',color=color,label='%s AUC = %0.3f'% (classifiers,roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.02,1.02])
    plt.ylim([-0.02,1.02])
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specifity')
fig = plt.gcf()
plt.subplot(1,2,1)

plot_roc(1,lr_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'LR','#fe5722','ROC curve of train set')
plot_roc(1,AB_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'AB','#03a8f3','ROC curve of train set')
plot_roc(1,mlp_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'MLP','#009587','ROC curve of train set')
plot_roc(1,Bag_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'BAG','#673ab6','ROC curve of train set')
plot_roc(1,gbm_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'GBM','#b5da3d','ROC curve of train set')
plot_roc(1,xgb_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'XGB','#3f51b4','ROC curve of train set')

plt.subplot(1,2,2)
plot_roc(1,lr_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'LR','#fe5722','ROC curve of test set')
plot_roc(1,AB_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'AB','#03a8f3','ROC curve of test set')
plot_roc(1,mlp_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'MLP','#009587','ROC curve of test set')
plot_roc(1,Bag_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'BAG','#673ab6','ROC curve of test set')
plot_roc(1,gbm_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'GBM','#b5da3d','ROC curve of test set')
plot_roc(1,xgb_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'XGB','#3f51b4','ROC curve of test set')
plt.show()
#%%ROC of Test
AB_model.fit(X_train, y_train)
logis_model.fit(X_train, y_train)
mlp.fit(X_train, y_train)
Bag_model.fit(X_train, y_train)
gbm_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
fpr_ab, tpr_ab, thresholds_ab = roc_curve(
    y_test, AB_model.predict_proba(X_test)[:, 1])
fpr_lr, tpr_lr, thresholds_lr = roc_curve(
    y_test, logis_model.predict_proba(X_test)[:, 1])
fpr_bag, tpr_bag, thresholds_bag = roc_curve(
    y_test, Bag_model.predict_proba(X_test)[:, 1])
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(
    y_test, mlp.predict_proba(X_test)[:, 1])
fpr_gbm, tpr_gbm, thresholds_gbm = roc_curve(
    y_test, gbm_model.predict_proba(X_test)[:, 1])
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(
    y_test, xgb_model.predict_proba(X_test)[:, 1])
roc_auc_ab = auc(fpr_ab, tpr_ab)
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_bag = auc(fpr_bag, tpr_bag)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
roc_auc_gbm = auc(fpr_gbm, tpr_gbm)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
plt.plot(fpr_lr, tpr_lr, label="ROC Curve LR; AUC = {:.3f}".format(roc_auc_lr))
plt.plot(fpr_ab, tpr_ab, label="ROC Curve AB; AUC = {:.3f}".format(roc_auc_ab))
plt.plot(fpr_bag, tpr_bag,
          label="ROC Curve BAG; AUC = {:.3f}".format(roc_auc_bag))
plt.plot(fpr_mlp, tpr_mlp,
          label="ROC Curve MLP; AUC = {:.3f}".format(roc_auc_mlp))
plt.plot(fpr_gbm, tpr_gbm,
          label="ROC Curve GBM; AUC = {:.3f}".format(roc_auc_gbm))
plt.plot(fpr_xgb, tpr_xgb,
          label="ROC Curve XGB; AUC = {:.3f}".format(roc_auc_xgb))
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label='Reference')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("1 - Specifity")
plt.ylabel("Sensitivity")
plt.title("ROC curve")
plt.legend(loc="lower right")
plt.show()
#%%Table 3
rf_score = AB_model.score(X_test, y_test)
lr_score = logis_model.score(X_test, y_test)
dt_score = Bag_model.score(X_test, y_test)
mlp_score = mlp.score(X_test, y_test)
gbm_score = gbm_model.score(X_test, y_test)
xgb_score = xgb_model.score(X_test, y_test)
rf_prob = AB_model.predict(X_test)
lr_prob = logis_model.predict(X_test)
dt_prob = Bag_model.predict(X_test)
mlp_prob = mlp.predict(X_test)
gbm_prob = gbm_model.predict(X_test)
xgb_prob = xgb_model.predict(X_test)
# 混淆矩阵
rf_cf = confusion_matrix(y_test, rf_prob)
lr_cf = confusion_matrix(y_test, lr_prob)
dt_cf = confusion_matrix(y_test, dt_prob)
mlp_cf = confusion_matrix(y_test, mlp_prob)
gbm_cf = confusion_matrix(y_test, gbm_prob)
xgb_cf = confusion_matrix(y_test, xgb_prob)
rf_cf
lr_cf
dt_cf
mlp_cf
gbm_cf
xgb_cf
TN_rf, FP_rf, FN_rf, TP_rf = confusion_matrix(y_test, rf_prob).ravel()
TN_lr, FP_lr, FN_lr, TP_lr = confusion_matrix(y_test, lr_prob).ravel()
TN_dt, FP_dt, FN_dt, TP_dt = confusion_matrix(y_test, dt_prob).ravel()
TN_mlp, FP_mlp, FN_mlp, TP_mlp = confusion_matrix(y_test, mlp_prob).ravel()
TN_gbm, FP_gbm, FN_gbm, TP_gbm = confusion_matrix(y_test, gbm_prob).ravel()
TN_xgb, FP_xgb, FN_xgb, TP_xgb = confusion_matrix(y_test, xgb_prob).ravel()
sen_rf, spc_rf = round(TP_rf/(TP_rf+FN_rf), 3), round(TN_rf/(FP_rf+TN_rf), 3)
sen_lr, spc_lr = round(TP_lr/(TP_lr+FN_lr), 3), round(TN_lr/(FP_lr+TN_lr), 3)
sen_dt, spc_dt = round(TP_dt/(TP_dt+FN_dt), 3), round(TN_dt/(FP_dt+TN_dt), 3)
sen_mlp, spc_mlp = round(TP_mlp/(TP_mlp+FN_mlp),
                          3), round(TN_mlp/(FP_mlp+TN_mlp), 3)
sen_gbm, spc_gbm = round(TP_gbm/(TP_gbm+FN_gbm),
                          3), round(TN_gbm/(FP_gbm+TN_gbm), 3)
sen_xgb, spc_xgb = round(TP_xgb/(TP_xgb+FN_xgb),
                          3), round(TN_xgb/(FP_xgb+TN_xgb), 3)
AB_f1 = f1_score(y_test, rf_prob, average='macro')
LR_f1 = f1_score(y_test, lr_prob, average='macro')
DT_f1 = f1_score(y_test, dt_prob, average='macro')
MLP_f1 = f1_score(y_test, mlp_prob, average='macro')
GBM_f1 = f1_score(y_test, gbm_prob, average='macro')
XGB_f1 = f1_score(y_test, xgb_prob, average='macro')

print("AB的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(AB_f1, roc_auc_ab, rf_score, sen_rf, spc_rf))
print("LR的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(LR_f1, roc_auc_lr, lr_score, sen_lr, spc_lr))
print("BAG的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(DT_f1, roc_auc_bag, dt_score, sen_dt, spc_dt))
print("MLP的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(MLP_f1, roc_auc_mlp, mlp_score, sen_mlp, spc_mlp))
print("GBM的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(GBM_f1, roc_auc_gbm, gbm_score, sen_gbm, spc_gbm))
print("XGB的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(XGB_f1, roc_auc_xgb, xgb_score, sen_xgb, spc_xgb))

#%% PR曲线
from sklearn.metrics import precision_recall_curve, average_precision_score
def ro_curve(k,y_pred, y_label, method_name,color,title):
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)    
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)    
    plt.figure(k)
    plt.plot(lr_recall, lr_precision, lw = 2, label= method_name + ' (Area = %0.3f)' % average_precision_score(y_label, y_pred),color=color)
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    plt.title(title)
    plt.legend(loc='upper right')
fig = plt.gcf()
#train 
plt.subplot(1,2,1)
ro_curve(1,lr_model.predict_proba(X_train)[:,1],y_train,'LR','red','Precision Recall Curve of train set')
ro_curve(1,AB_model.predict_proba(X_train)[:,1],y_train,'AB','blue','Precision Recall Curve of train set')
ro_curve(1,mlp_model.predict_proba(X_train)[:,1],y_train,'MLP','green','Precision Recall Curve of train set')
ro_curve(1,Bag_model.predict_proba(X_train)[:,1],y_train,'BAG','m','Precision Recall Curve of train set')
ro_curve(1,gbm_model.predict_proba(X_train)[:,1],y_train,'GBM','tomato','Precision Recall Curve of train set')
ro_curve(1,xgb_model.predict_proba(X_train)[:,1],y_train,'XGB','darkblue','Precision Recall Curve of train set')

#test
plt.subplot(1,2,2)
ro_curve(1,lr_model.predict_proba(X_test)[:,1],y_test,'LR','red','Precision Recall Curve of test set')
ro_curve(1,AB_model.predict_proba(X_test)[:,1],y_test,'AB','blue','Precision Recall Curve of test set')
ro_curve(1,mlp.predict_proba(X_test)[:,1],y_test,'MLP','green','Precision Recall Curve of test set')
ro_curve(1,Bag_model.predict_proba(X_test)[:,1],y_test,'BAG','m','Precision Recall Curve of test set')
ro_curve(1,gbm_model.predict_proba(X_test)[:,1],y_test,'GBM','tomato','Precision Recall Curve of test set')
ro_curve(1,xgb_model.predict_proba(X_test)[:,1],y_test,'XGB','darkblue','Precision Recall Curve of test set')
plt.show()
#%%校准曲线

rf_prob = AB_model.predict(X_test)
lr_prob = logis_model.predict(X_test)
dt_prob = Bag_model.predict(X_test)
mlp_prob = mlp.predict(X_test)
gbm_prob = gbm_model.predict(X_test)
xgb_prob = xgb_model.predict(X_test)
plt.rcParams["axes.grid"] = False
sns.set()
from sklearn.calibration import calibration_curve
def calibration_curve_1(k,y_pred,y_true,method_name,color,title):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=5)
    plt.figure(k)
    plt.plot(prob_pred,prob_true,color=color,label='%s calibration_curve'%method_name,marker='s')
    plt.plot([i/100 for i in range(0,100)],[i/100 for i in range(0,100)],color='black',linestyle='--')
    plt.xlim(-0.02,1.02,0.2)
    plt.ylim(-0.02,1.02,0.2)
    plt.xlabel('y_preds')
    plt.ylabel('y_real')
    plt.title(title)
    plt.legend(loc='lower right')
plt.subplot(1,2,1)
calibration_curve_1(1,logis_model.predict_proba(X_train)[:,1],y_train,'LR','red','Calibration curve of train set')
calibration_curve_1(1,AB_model.predict_proba(X_train)[:,1],y_train,'AB','blue','Calibration curve of train set')
calibration_curve_1(1,mlp.predict_proba(X_train)[:,1],y_train,'MLP','green','Calibration curve of train set')
calibration_curve_1(1,Bag_model.predict_proba(X_train)[:,1],y_train,'BAG','m','Calibration curve of train set')
calibration_curve_1(1,gbm_model.predict_proba(X_train)[:,1],y_train,'GBM','tomato','Calibration curve of train set')
calibration_curve_1(1,xgb_model.predict_proba(X_train)[:,1],y_train,'XGB','darkblue','Calibration curve of train set')
plt.subplot(1,2,2)
calibration_curve_1(1,logis_model.predict_proba(X_test)[:,1],y_test,'LR','red','Calibration curve of test set')
calibration_curve_1(1,AB_model.predict_proba(X_test)[:,1],y_test,'AB','blue','Calibration curve of test set')
calibration_curve_1(1,mlp.predict_proba(X_test)[:,1],y_test,'MLP','green','Calibration curve of test set')
calibration_curve_1(1,Bag_model.predict_proba(X_test)[:,1],y_test,'BAG','m','Calibration curve of test set')
calibration_curve_1(1,gbm_model.predict_proba(X_test)[:,1],y_test,'GBM','tomato','Calibration curve of test set')
calibration_curve_1(1,xgb_model.predict_proba(X_test)[:,1],y_test,'XGB','darkblue','Calibration curve of test set')
plt.show()
#%%绘制混淆矩阵train
xgb_prob_train = xgb_model.predict(X_train)
cm = confusion_matrix(y_train, xgb_prob_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low risk', 'High risk'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of XGB in train set")
plt.show()
#%%绘制混淆矩阵test
xgb_prob_test = xgb_model.predict(X_test)
cm = confusion_matrix(y_test, xgb_prob_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low risk', 'High risk'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of XGB in test set")
plt.show()
#%%最佳机器学习模型5折交叉验证 XGB
sns.set()
cv = StratifiedKFold(n_splits=5)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
for i, (train, test) in enumerate(cv.split(X, y)):
    xgb_model.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        xgb_model,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2,
        color="r", label="Reference", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="XGB Receiver operating characteristic ",
)
ax.legend(loc="lower right")
plt.xlabel("1 - Specifity")
plt.ylabel("Sensitivity")
plt.show()

#%%特征重要性
feature_data = train_data[features]
sns.set()
explainer = shap.Explainer(xgb_model, feature_data)
shap_values = explainer.shap_values(feature_data)  # 传入特征矩阵X，计算SHAP值
                
#Feature importances1 and 2
a = 72
shap.initjs()
plot1 = shap.force_plot(explainer.expected_value,
                shap_values[a, :], 
                feature_data.iloc[a, :], 
                figsize=(15, 5),
                # link = "logit",
                matplotlib=True,
                out_names = "Output value")
# #Feature importances
sns.set()
shap.summary_plot(shap_values, 
                  feature_data,
                  plot_type="violin", 
                  max_display=10,
                  color='#3d5afe',
                  title='Feature importance')
#柱状图
shap.summary_plot(shap_values, feature_data, plot_type="bar")


#%%保存最佳模型的输出概率值
prob_train = pd.DataFrame(xgb_model.predict_proba(np.array(train_data[features])))
prob_train.to_csv("/Users/mac/Desktop/吴世楠文件夹/肾癌翻修/renal_cancer_ml (2)/xgb_pro_train.csv", index=False, sep=',')

prob_test = pd.DataFrame(xgb_model.predict_proba(np.array(test_data[features])))
prob_test.to_csv("/Users/mac/Desktop/吴世楠文件夹/肾癌翻修/renal_cancer_ml (2)/xgb_pro_test.csv", index=False, sep=',')

