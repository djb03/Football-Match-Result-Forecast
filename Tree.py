# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:27:32 2021

@author: calvin
"""


#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from numpy import arange
import csv

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from scipy.stats import uniform, randint
import xgboost as xgb
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance

import graphviz
#%%
data=pd.read_excel("data_tree.xlsx") #please add your own document path
#%%
data_t=data.loc[data.season.isin(['2013/2014', '2014/2015'])]
data_v=data.loc[data.season.isin(['2015/2016'])]
#%%
GAP1=['GAP_H_Shoton','GAP_A_Shoton','GAP_H_Shotoff','GAP_A_Shotoff','GAP_H_Cross', 'GAP_A_Cross']
GAP2=['GAP_H_Shot', 'GAP_A_Shot','GAP_H_Cross', 'GAP_A_Cross']
Stage=['stage']
FIFA=['CB_POW_H', 'CB_MEN_H', 'CB_SKI_H', 'CB_MOV_H',
       'CB_ATT_H', 'CB_DEF_H', 'CM_POW_H', 'CM_MEN_H', 'CM_SKI_H', 'CM_MOV_H',
       'CM_ATT_H', 'CM_DEF_H', 'GK_POW_H', 'GK_MEN_H',  'GK_MOV_H',
         'GK_GOK_H', 'ST_POW_H', 'ST_MEN_H', 'ST_SKI_H',
       'ST_MOV_H', 'ST_ATT_H', 'ST_DEF_H', 'CB_POW_A', 'CB_MEN_A', 'CB_SKI_A',
       'CB_MOV_A', 'CB_ATT_A', 'CB_DEF_A', 'CM_POW_A', 'CM_MEN_A', 'CM_SKI_A',
       'CM_MOV_A', 'CM_ATT_A', 'CM_DEF_A', 'GK_POW_A', 'GK_MEN_A', 
       'GK_MOV_A',  'GK_GOK_A', 'ST_POW_A', 'ST_MEN_A',
       'ST_SKI_A', 'ST_MOV_A', 'ST_ATT_A', 'ST_DEF_A']
odds=['B365_home_win_Prob']

LR1=['LR_h_shoton', 'LR_h_shotoff',  'LR_h_cross', 'LR_a_shoton',
       'LR_a_shotoff',  'LR_a_cross']
LR2=[ 'LR_h_shot', 'LR_h_cross',  'LR_a_shot', 'LR_a_cross']

result=['home_result']
#%%
x11=data_t[GAP1+FIFA+Stage]  #GAP, FIFA, stage and 3stat
x12=data_t[GAP2+FIFA+Stage]  #GAP FIFA, stage and 2stat

x21=data_t[LR1+Stage]  # LR, stage and 3stat
x22=data_t[LR2+Stage]  # LR, stage and 2stat

x3=data_t[FIFA] 
y=data_t[result]   # result from 2013-2014

#%% 2015 data
t11=data_v[GAP1+FIFA+Stage]
t12=data_v[GAP2+FIFA+Stage]


t21=data_v[LR1+Stage] 
t22=data_v[LR2+Stage]

t3=data_v[FIFA]
v=data_v[result]
#%% 2015 odds
odds=data_v[odds]
#%%
odd_home_win=np.where(odds >= 0.5, 1, 0)
#%%
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,tree_method='gpu_hist', gpu_id=0)

params = {    
    "gamma": arange(0,0.5,0.1),
    "learning_rate": arange(0,1,0.1), # default 0.1 
    "max_depth": arange(1,5,1), # default 3
    "n_estimators": [100,200,300] # default 100 
}

xgb3 = GridSearchCV(xgb_model, params, cv=[(slice(None), slice(None))]
                      , n_jobs=-1)

m3=xgb3.fit(x3, y)

print("Best: %f using %s" % (m3.best_score_, m3.best_params_))

p3 = m3.predict_proba(t3)
#%%
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,tree_method='gpu_hist', gpu_id=0)

params = {    
    "gamma": arange(0,0.5,0.1),
    "learning_rate": arange(0,1,0.1), # default 0.1 
    "max_depth": arange(1,5,1), # default 3
    "n_estimators": [100,200,300] # default 100 
}

xgb11 = GridSearchCV(xgb_model, params, cv=[(slice(None), slice(None))]
                      , n_jobs=-1)

m11=xgb11.fit(x11, y)

print("Best: %f using %s" % (m11.best_score_, m11.best_params_))

p11 = m11.predict_proba(t11)


xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,tree_method='gpu_hist', gpu_id=0)

params = {    
    "gamma": arange(0,0.5,0.1),
    "learning_rate": arange(0,1,0.1), # default 0.1 
    "max_depth": arange(1,5,1), # default 3
    "n_estimators": [100,200,300] # default 100 
}

xgb12 = GridSearchCV(xgb_model, params, cv=[(slice(None), slice(None))]
                      , n_jobs=-1)

m12=xgb12.fit(x12, y)

print("Best: %f using %s" % (m12.best_score_, m12.best_params_))

p12 = m12.predict_proba(t12)

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,tree_method='gpu_hist', gpu_id=0)

params = {    
    "gamma": arange(0,0.5,0.1),
    "learning_rate": arange(0,1,0.1), # default 0.1 
    "max_depth": arange(1,5,1), # default 3
    "n_estimators": [100,200,300] # default 100 
}

xgb21 = GridSearchCV(xgb_model, params, cv=[(slice(None), slice(None))]
                      , n_jobs=-1)

m21=xgb21.fit(x21, y)

print("Best: %f using %s" % (m21.best_score_, m21.best_params_))

p21 = m21.predict_proba(t21)


xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,tree_method='gpu_hist', gpu_id=0)

params = {    
    "gamma": arange(0,0.5,0.1),
    "learning_rate": arange(0,1,0.1), # default 0.1 
    "max_depth": arange(1,5,1), # default 3
    "n_estimators": [100,200,300] # default 100 
}

xgb22 = GridSearchCV(xgb_model, params, cv=[(slice(None), slice(None))]
                      , n_jobs=-1)

m22=xgb22.fit(x22, y)

print("Best: %f using %s" % (m22.best_score_, m22.best_params_))

p22 = m22.predict_proba(t22)

#%%
b11=brier_score_loss(v,p11[:,1])
b12=brier_score_loss(v,p12[:,1])
b21=brier_score_loss(v,p21[:,1])
b22=brier_score_loss(v,p22[:,1])
b3=brier_score_loss(v,odds)


f11=f1_score(v,m11.predict(t11))
f12=f1_score(v,m12.predict(t12))
f21=f1_score(v,m21.predict(t21))
f22=f1_score(v,m22.predict(t22))
f3=f1_score(v,odd_home_win)

l11=log_loss(v,p11[:,1])
l12=log_loss(v,p12[:,1])
l21=log_loss(v,p21[:,1])
l22=log_loss(v,p22[:,1])
l3=log_loss(v,odds)

r11=roc_auc_score(v,m11.predict(t11))
r12=roc_auc_score(v,m12.predict(t12))
r21=roc_auc_score(v,m21.predict(t21))
r22=roc_auc_score(v,m22.predict(t22))
r3=roc_auc_score(v,odd_home_win)
#%%
bf=brier_score_loss(v,p3[:,1])
ff=f1_score(v,m3.predict(t3))
lf=log_loss(v,p3[:,1])
rf=roc_auc_score(v,m3.predict(t3))
#%%
score=pd.DataFrame([[b11,l11,f11,r11],[b12,l12,f12,r12],[b21,l21,f21,r21],
                    [b22,l22,f22,r22],[b3,l3,f3,r3]],
                   columns=['brier_score','log_loss','f1_score',"roc_auc_score"])
#%%
score.to_excel('tree_score.xlsx')
#%% save model
m11.best_estimator_.save_model("m11.txt")
m12.best_estimator_.save_model("m12.txt")
m21.best_estimator_.save_model("m21.txt")
m22.best_estimator_.save_model("m22.txt")
#%% load model
m11= xgb.Booster()
m12= xgb.Booster()
m21= xgb.Booster()
m22= xgb.Booster()
m11.load_model("m11.txt")
m12.load_model("m12.txt")
m21.load_model("m21.txt")
m22.load_model("m22.txt")
#%% prediction
p11 = m11.predict(t11)
#%%
p12 = m12.predict_proba(t12)
p21 = m21.predict_proba(t21)
p22 = m22.predict_proba(t22)
#%% Features plot 
fig, ax = plt.subplots(figsize=(5, 10))
GAP1_fig=xgb.plot_importance(m11.best_estimator_,ax=ax)

#%%
fig, ax = plt.subplots(figsize=(5, 10))
GAP2_fig=xgb.plot_importance(m12.best_estimator_,ax=ax)

#%%
LR1_fig=xgb.plot_importance(m21.best_estimator_,)
LR2_fig=xgb.plot_importance(m22.best_estimator_,)
#%%
fig, ax = plt.subplots(figsize=(5, 10))
FIFA_fig=xgb.plot_importance(m3.best_estimator_,ax=ax)
#%% Features plot with saved model
fig, ax = plt.subplots(figsize=(5, 10))
xgb.plot_importance(m11,ax=ax)
#%%
xgb.plot_importance(m12,)
xgb.plot_importance(m21,)
xgb.plot_importance(m22,)


#%%
result = pd.DataFrame(p11)
result["stage"]=data_v["stage"]

#%% 
result1=pd.concat([pd.DataFrame(p11), data_v["stage"].reset_index()], axis=1, ignore_index=True)
# stage 1 to 6 row 0 to 44 stage 33 to 38 302-360
result2=result1[45:302]
#%% remove stage 1 to 6 and stage 33 to 38 error 
# m11 GAP 3 stat, m12 GAP 2 stat
var11=m11.predict(t11)
var12=m12.predict(t12)

var21=m21.predict(t21)
var22=m22.predict(t22)
var3=m3.predict(t3)
#%%
b11=brier_score_loss(v[45:302],p11[45:302,1])
b12=brier_score_loss(v[45:302],p12[45:302,1])
b21=brier_score_loss(v[45:302],p21[45:302,1])
b22=brier_score_loss(v[45:302],p22[45:302,1])
b3=brier_score_loss(v[45:302],odds[45:302])

f11=f1_score(v[45:302],var11[45:302])
f12=f1_score(v[45:302],var12[45:302])
f21=f1_score(v[45:302],var21[45:302])
f22=f1_score(v[45:302],var22[45:302])
f3=f1_score(v[45:302],odd_home_win[45:302])

l11=log_loss(v[45:302],p11[45:302,1])
l12=log_loss(v[45:302],p12[45:302,1])
l21=log_loss(v[45:302],p21[45:302,1])
l22=log_loss(v[45:302],p22[45:302,1])
l3=log_loss(v[45:302],odds[45:302])

r11=roc_auc_score(v[45:302],var11[45:302])
r12=roc_auc_score(v[45:302],var12[45:302])
r21=roc_auc_score(v[45:302],var21[45:302])
r22=roc_auc_score(v[45:302],var22[45:302])
r3=roc_auc_score(v[45:302],odd_home_win[45:302])

bf=brier_score_loss(v[45:302],p3[45:302,1])
ff=f1_score(v[45:302],var3[45:302])
lf=log_loss(v[45:302],p3[45:302,1])
rf=roc_auc_score(v[45:302],var3[45:302])
#%%
score = {'brier_score': [b11,b12,b21,b22,b3,bf], 'log_loss': [l11,l12,l21,l22,l3,lf],
         "f1_score":[f11,f12,f21,f22,f3,ff],"roc_auc_score":[r11,r12,r21,r22,r3,rf] }

score=pd.DataFrame(score)