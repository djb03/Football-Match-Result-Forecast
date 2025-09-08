# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:13:23 2021

@author: calvin
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from numpy import arange
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
from sklearn import linear_model
from scipy import stats
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
#%%
data=pd.read_excel("data_LR.xlsx") #please add your own document path
data=data.drop('Unnamed: 0', axis=1) #drop repeated index
data=data.drop(['GK_SKI_H','GK_ATT_H','GK_DEF_H','GK_SKI_A','GK_ATT_A', 'GK_DEF_A'], axis=1) #drop unwanted FIFA rating
data=data.drop(['home_team_goal', 'away_team_goal'], axis=1) #drop unwanted Match Statistics
data["stage"] = pd.to_numeric(data["stage"]) # turn stage from str to int64
data=data.sort_values(by=["season","stage"]) #sort by "season" then "stage"
data["home_shot"]=data['home_shoton']+data['home_shotoff']
data["away_shot"]=data['away_shoton']+data['away_shotoff']

#%%
data1=data
#%%normalization divide
    
nogk=['CB_POW_H', 'CB_MEN_H', 'CB_SKI_H', 'CB_MOV_H',
       'CB_ATT_H', 'CB_DEF_H', 'CM_POW_H', 'CM_MEN_H', 'CM_SKI_H', 'CM_MOV_H',
       'CM_ATT_H', 'CM_DEF_H',  'ST_POW_H', 'ST_MEN_H', 'ST_SKI_H',
       'ST_MOV_H', 'ST_ATT_H', 'ST_DEF_H', 'CB_POW_A', 'CB_MEN_A', 'CB_SKI_A',
       'CB_MOV_A', 'CB_ATT_A', 'CB_DEF_A', 'CM_POW_A', 'CM_MEN_A', 'CM_SKI_A',
       'CM_MOV_A', 'CM_ATT_A', 'CM_DEF_A',  'ST_POW_A', 'ST_MEN_A',
       'ST_SKI_A', 'ST_MOV_A', 'ST_ATT_A', 'ST_DEF_A']
yesgk=['GK_POW_A', 'GK_MEN_A', 
       'GK_MOV_A',  'GK_GOK_A','GK_POW_H', 'GK_MEN_H',  'GK_MOV_H',
         'GK_GOK_H']
#for i in nogk:
#    data[i]=data[i]/500
#for i in yesgk:
#    data[i]=data[i]/100
#%% Splite Training, Validation adn Testing Set
t1=data.loc[data.season.isin(['2011/2012', '2012/2013'])]
t2=data.loc[data.season.isin(['2012/2013', '2013/2014'])]
t3=data.loc[data.season.isin(['2013/2014', '2014/2015'])]
v1=data.loc[data.season.isin(['2013/2014'])]
v2=data.loc[data.season.isin(['2014/2015'])]
v3=data.loc[data.season.isin(['2015/2016'])]
a1=data.loc[data.season.isin([ '2012/2013'])]
a2=data.loc[data.season.isin([ '2013/2014'])]

#%%
FIFA=['CB_POW_H', 'CB_MEN_H', 'CB_SKI_H', 'CB_MOV_H',
       'CB_ATT_H', 'CB_DEF_H', 'CM_POW_H', 'CM_MEN_H', 'CM_SKI_H', 'CM_MOV_H',
       'CM_ATT_H', 'CM_DEF_H', 'GK_POW_H', 'GK_MEN_H',  'GK_MOV_H',
         'GK_GOK_H', 'ST_POW_H', 'ST_MEN_H', 'ST_SKI_H',
       'ST_MOV_H', 'ST_ATT_H', 'ST_DEF_H', 'CB_POW_A', 'CB_MEN_A', 'CB_SKI_A',
       'CB_MOV_A', 'CB_ATT_A', 'CB_DEF_A', 'CM_POW_A', 'CM_MEN_A', 'CM_SKI_A',
       'CM_MOV_A', 'CM_ATT_A', 'CM_DEF_A', 'GK_POW_A', 'GK_MEN_A', 
       'GK_MOV_A',  'GK_GOK_A', 'ST_POW_A', 'ST_MEN_A',
       'ST_SKI_A', 'ST_MOV_A', 'ST_ATT_A', 'ST_DEF_A']
MS=['home_shoton', 'away_shoton', 'home_shotoff', 'away_shotoff',\
       'home_cross_sum', 'away_cross_sum']
odds=['B365_home_win_Prob']
result=['home_result']

#%% t1 
t1_x=t1[FIFA]
t1_y_h_shoton=t1['home_shoton']
t1_y_a_shoton=t1['away_shoton']
t1_y_h_shotoff=t1['home_shotoff']
t1_y_a_shotoff=t1['away_shotoff']
t1_y_h_shot=t1['home_shot']
t1_y_a_shot=t1['away_shot']
t1_y_h_cross=t1['home_cross_sum']
t1_y_a_cross=t1['away_cross_sum']

v1_x=v1[FIFA]
v1_y_h_shoton=v1['home_shoton']
v1_y_a_shoton=v1['away_shoton']
v1_y_h_shotoff=v1['home_shotoff']
v1_y_a_shotoff=v1['away_shotoff']
v1_y_h_shot=v1['home_shot']
v1_y_a_shot=v1['away_shot']
v1_y_h_cross=v1['home_cross_sum']
v1_y_a_cross=v1['away_cross_sum']

t2_x=t2[FIFA]
t2_y_h_shoton=t2['home_shoton']
t2_y_a_shoton=t2['away_shoton']
t2_y_h_shotoff=t2['home_shotoff']
t2_y_a_shotoff=t2['away_shotoff']
t2_y_h_shot=t2['home_shot']
t2_y_a_shot=t2['away_shot']
t2_y_h_cross=t2['home_cross_sum']
t2_y_a_cross=t2['away_cross_sum']

v2_x=v2[FIFA]
v2_y_h_shoton=v2['home_shoton']
v2_y_a_shoton=v2['away_shoton']
v2_y_h_shotoff=v2['home_shotoff']
v2_y_a_shotoff=v2['away_shotoff']
v2_y_h_shot=v2['home_shot']
v2_y_a_shot=v2['away_shot']
v2_y_h_cross=v2['home_cross_sum']
v2_y_a_cross=v2['away_cross_sum']

t3_x=t3[FIFA]
t3_y_h_shoton=t3['home_shoton']
t3_y_a_shoton=t3['away_shoton']
t3_y_h_shotoff=t3['home_shotoff']
t3_y_a_shotoff=t3['away_shotoff']
t3_y_h_shot=t3['home_shot']
t3_y_a_shot=t3['away_shot']
t3_y_h_cross=t3['home_cross_sum']
t3_y_a_cross=t3['away_cross_sum']

v3_x=v3[FIFA]
v3_y_h_shoton=v3['home_shoton']
v3_y_a_shoton=v3['away_shoton']
v3_y_h_shotoff=v3['home_shotoff']
v3_y_a_shotoff=v3['away_shotoff']
v3_y_h_shot=v3['home_shot']
v3_y_a_shot=v3['away_shot']
v3_y_h_cross=v3['home_cross_sum']
v3_y_a_cross=v3['away_cross_sum']

a1_x=a1[FIFA]
a1_y_h_shoton=a1['home_shoton']
a1_y_a_shoton=a1['away_shoton']
a1_y_h_shotoff=a1['home_shotoff']
a1_y_a_shotoff=a1['away_shotoff']
a1_y_h_shot=a1['home_shot']
a1_y_a_shot=a1['away_shot']
a1_y_h_cross=a1['home_cross_sum']
a1_y_a_cross=a1['away_cross_sum']

a2_x=a2[FIFA]
a2_y_h_shoton=a2['home_shoton']
a2_y_a_shoton=a2['away_shoton']
a2_y_h_shotoff=a2['home_shotoff']
a2_y_a_shotoff=a2['away_shotoff']
a2_y_h_shot=a2['home_shot']
a2_y_a_shot=a2['away_shot']
a2_y_h_cross=a2['home_cross_sum']
a2_y_a_cross=a2['away_cross_sum']
#%%
EN_t1_h_shoton = LinearRegression()
EN_t1_h_shoton.fit(t1_x, t1_y_h_shoton)
EN_t2_h_shoton = LinearRegression()
EN_t2_h_shoton.fit(t2_x, t2_y_h_shoton)
#EN_t3_h_shoton = LinearRegression()
#EN_t3_h_shoton.fit(t3_x, t3_y_h_shoton)

EN_t1_h_shotoff = LinearRegression()
EN_t1_h_shotoff.fit(t1_x, t1_y_h_shotoff)
EN_t2_h_shotoff = LinearRegression()
EN_t2_h_shotoff.fit(t2_x, t2_y_h_shotoff)
#EN_t3_h_shotoff = LinearRegression()
#EN_t3_h_shotoff.fit(t3_x, t3_y_h_shotoff)

EN_t1_h_shot = LinearRegression()
EN_t1_h_shot.fit(t1_x, t1_y_h_shot)
EN_t2_h_shot = LinearRegression()
EN_t2_h_shot.fit(t2_x, t2_y_h_shot)
#EN_t3_h_shot = LinearRegression()
#EN_t3_h_shot.fit(t3_x, t3_y_h_shot)

EN_t1_h_cross = LinearRegression()
EN_t1_h_cross.fit(t1_x, t1_y_h_cross)
EN_t2_h_cross = LinearRegression()
EN_t2_h_cross.fit(t2_x, t2_y_h_cross)
#EN_t3_h_cross = LinearRegression()
#EN_t3_h_cross.fit(t3_x, t3_y_h_cross)

EN_t1_a_shoton = LinearRegression()
EN_t1_a_shoton.fit(t1_x, t1_y_a_shoton)
EN_t2_a_shoton = LinearRegression()
EN_t2_a_shoton.fit(t2_x, t2_y_a_shoton)
#EN_t3_a_shoton = LinearRegression()
#EN_t3_a_shoton.fit(t3_x, t3_y_a_shoton)

EN_t1_a_shotoff = LinearRegression()
EN_t1_a_shotoff.fit(t1_x, t1_y_a_shotoff)
EN_t2_a_shotoff = LinearRegression()
EN_t2_a_shotoff.fit(t2_x, t2_y_a_shotoff)
#EN_t3_a_shotoff = LinearRegression()
#EN_t3_a_shotoff.fit(t3_x, t3_y_a_shotoff)

EN_t1_a_shot = LinearRegression()
EN_t1_a_shot.fit(t1_x, t1_y_a_shot)
EN_t2_a_shot = LinearRegression()
EN_t2_a_shot.fit(t2_x, t2_y_a_shot)
#EN_t3_a_shot = LinearRegression()
#EN_t3_a_shot.fit(t3_x, t3_y_a_shot)

EN_t1_a_cross = LinearRegression()
EN_t1_a_cross.fit(t1_x, t1_y_a_cross)
EN_t2_a_cross = LinearRegression()
EN_t2_a_cross.fit(t2_x, t2_y_a_cross)
#EN_t3_a_cross = LinearRegression()
#EN_t3_a_cross.fit(t3_x, t3_y_a_cross)
#%%
EN_t1_h_shoton = LinearRegression()
EN_t1_h_shoton.fit(t1_x, t1_y_h_shoton)

#%%
coef=EN_t1_h_shoton. coef_

#%% float to dec
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
#%%
X = sm.add_constant(t1_x)
model = sm.OLS( t1_y_h_shoton,t1_x)
results = model.fit()
#%%
# VIF dataframe
X =data[FIFA]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)
#%%
print(results.params)
#%%
print(results.summary())
#%% Prediction
p_t1_h_shoton=EN_t1_h_shoton.predict(v1_x)
p_t2_h_shoton=EN_t2_h_shoton.predict(v2_x)


p_t1_h_shotoff=EN_t1_h_shotoff.predict(v1_x)
p_t2_h_shotoff=EN_t2_h_shotoff.predict(v2_x)


p_t1_h_shot=EN_t1_h_shot.predict(v1_x)
p_t2_h_shot=EN_t2_h_shot.predict(v2_x)


p_t1_h_cross=EN_t1_h_cross.predict(v1_x)
p_t2_h_cross=EN_t2_h_cross.predict(v2_x)


p_t1_a_shoton=EN_t1_a_shoton.predict(v1_x)
p_t2_a_shoton=EN_t2_a_shoton.predict(v2_x)

p_t1_a_shotoff=EN_t1_a_shotoff.predict(v1_x)
p_t2_a_shotoff=EN_t2_a_shotoff.predict(v2_x)


p_t1_a_shot=EN_t1_a_shot.predict(v1_x)
p_t2_a_shot=EN_t2_a_shot.predict(v2_x)


p_t1_a_cross=EN_t1_a_cross.predict(v1_x)
p_t2_a_cross=EN_t2_a_cross.predict(v2_x)

#%%
p_a1_h_shoton=EN_t1_h_shoton.predict(a1_x)
p_a2_h_shoton=EN_t2_h_shoton.predict(a2_x)


p_a1_h_shotoff=EN_t1_h_shotoff.predict(a1_x)
p_a2_h_shotoff=EN_t2_h_shotoff.predict(a2_x)


p_a1_h_shot=EN_t1_h_shot.predict(a1_x)
p_a2_h_shot=EN_t2_h_shot.predict(a2_x)


p_a1_h_cross=EN_t1_h_cross.predict(a1_x)
p_a2_h_cross=EN_t2_h_cross.predict(a2_x)


p_a1_a_shoton=EN_t1_a_shoton.predict(a1_x)
p_a2_a_shoton=EN_t2_a_shoton.predict(a2_x)

p_a1_a_shotoff=EN_t1_a_shotoff.predict(a1_x)
p_a2_a_shotoff=EN_t2_a_shotoff.predict(a2_x)


p_a1_a_shot=EN_t1_a_shot.predict(a1_x)
p_a2_a_shot=EN_t2_a_shot.predict(a2_x)


p_a1_a_cross=EN_t1_a_cross.predict(a1_x)
p_a2_a_cross=EN_t2_a_cross.predict(a2_x)

#%%
data2013 = pd.DataFrame(zip(p_t1_h_shoton, p_t1_h_shotoff, p_t1_h_shot,p_t1_h_cross,
                            p_t1_a_shoton, p_t1_a_shotoff, p_t1_a_shot,p_t1_a_cross), columns=[
                                'h_shoton','h_shotoff','h_shot','h_cross',
                                'a_shoton','a_shotoff','a_shot','a_cross'])
data2014 = pd.DataFrame(zip(p_t2_h_shoton, p_t2_h_shotoff, p_t2_h_shot,p_t2_h_cross,
                            p_t2_a_shoton, p_t2_a_shotoff, p_t2_a_shot,p_t2_a_cross), columns=[
                                'h_shoton','h_shotoff','h_shot','h_cross',
                                'a_shoton','a_shotoff','a_shot','a_cross'])
# data2015 = pd.DataFrame(zip(p_t3_h_shoton, p_t3_h_shotoff, p_t3_h_shot,p_t3_h_cross,
#                             p_t3_a_shoton, p_t3_a_shotoff, p_t3_a_shot,p_t3_a_cross), columns=[
#                                 'h_shoton','h_shotoff','h_shot','h_cross',
#                                 'a_shoton','a_shotoff','a_shot','a_cross'])                               
#%% Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('LR_Prediction.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
data2013.to_excel(writer, sheet_name='Sheet1')
data2014.to_excel(writer, sheet_name='Sheet2')
# data2015.to_excel(writer, sheet_name='Sheet3')

# Close the Pandas Excel writer and output the Excel file.
writer.close()                          
#%%

# mse_t1_h_shoton=mean_squared_error(p_t1_h_shoton,v1_y_h_shoton)
# mse_t2_h_shoton=mean_squared_error(p_t2_h_shoton,v2_y_h_shoton)
# # mse_t3_h_shoton=mean_squared_error(p_t3_h_shoton,v3_y_h_shoton)

# mse_t1_h_shotoff=mean_squared_error(p_t1_h_shotoff,v1_y_h_shotoff)
# mse_t2_h_shotoff=mean_squared_error(p_t2_h_shotoff,v2_y_h_shotoff)
# # mse_t3_h_shotoff=mean_squared_error(p_t3_h_shotoff,v3_y_h_shotoff)

# mse_t1_h_shot=mean_squared_error(p_t1_h_shot,v1_y_h_shot)
# mse_t2_h_shot=mean_squared_error(p_t2_h_shot,v2_y_h_shot)
# # mse_t3_h_shot=mean_squared_error(p_t3_h_shot,v3_y_h_shot)

# mse_t1_h_cross=mean_squared_error(p_t1_h_cross,v1_y_h_cross)
# mse_t2_h_cross=mean_squared_error(p_t2_h_cross,v2_y_h_cross)
# # mse_t3_h_cross=mean_squared_error(p_t3_h_cross,v3_y_h_cross)

# mse_t1_a_shoton=mean_squared_error(p_t1_a_shoton,v1_y_a_shoton)
# mse_t2_a_shoton=mean_squared_error(p_t2_a_shoton,v2_y_a_shoton)
# # mse_t3_a_shoton=mean_squared_error(p_t3_a_shoton,v3_y_a_shoton)

# mse_t1_a_shotoff=mean_squared_error(p_t1_a_shotoff,v1_y_a_shotoff)
# mse_t2_a_shotoff=mean_squared_error(p_t2_a_shotoff,v2_y_a_shotoff)
# # mse_t3_a_shotoff=mean_squared_error(p_t3_a_shotoff,v3_y_a_shotoff)

# mse_t1_a_shot=mean_squared_error(p_t1_a_shot,v1_y_a_shot)
# mse_t2_a_shot=mean_squared_error(p_t2_a_shot,v2_y_a_shot)
# # mse_t3_a_shot=mean_squared_error(p_t3_a_shot,v3_y_a_shot)

# mse_t1_a_cross=mean_squared_error(p_t1_a_cross,v1_y_a_cross)
# mse_t2_a_cross=mean_squared_error(p_t2_a_cross,v2_y_a_cross)
# # mse_t3_a_cross=mean_squared_error(p_t3_a_cross,v3_y_a_cross)

# mae_t1_h_shoton=mean_absolute_error(p_t1_h_shoton,v1_y_h_shoton)
# mae_t2_h_shoton=mean_absolute_error(p_t2_h_shoton,v2_y_h_shoton)
# # mae_t3_h_shoton=mean_absolute_error(p_t3_h_shoton,v3_y_h_shoton)

# mae_t1_h_shotoff=mean_absolute_error(p_t1_h_shotoff,v1_y_h_shotoff)
# mae_t2_h_shotoff=mean_absolute_error(p_t2_h_shotoff,v2_y_h_shotoff)
# # mae_t3_h_shotoff=mean_absolute_error(p_t3_h_shotoff,v3_y_h_shotoff)

# mae_t1_h_shot=mean_absolute_error(p_t1_h_shot,v1_y_h_shot)
# mae_t2_h_shot=mean_absolute_error(p_t2_h_shot,v2_y_h_shot)
# # mae_t3_h_shot=mean_absolute_error(p_t3_h_shot,v3_y_h_shot)

# mae_t1_h_cross=mean_absolute_error(p_t1_h_cross,v1_y_h_cross)
# mae_t2_h_cross=mean_absolute_error(p_t2_h_cross,v2_y_h_cross)
# # mae_t3_h_cross=mean_absolute_error(p_t3_h_cross,v3_y_h_cross)

# mae_t1_a_shoton=mean_absolute_error(p_t1_a_shoton,v1_y_a_shoton)
# mae_t2_a_shoton=mean_absolute_error(p_t2_a_shoton,v2_y_a_shoton)
# # mae_t3_a_shoton=mean_absolute_error(p_t3_a_shoton,v3_y_a_shoton)

# mae_t1_a_shotoff=mean_absolute_error(p_t1_a_shotoff,v1_y_a_shotoff)
# mae_t2_a_shotoff=mean_absolute_error(p_t2_a_shotoff,v2_y_a_shotoff)
# # mae_t3_a_shotoff=mean_absolute_error(p_t3_a_shotoff,v3_y_a_shotoff)

# mae_t1_a_shot=mean_absolute_error(p_t1_a_shot,v1_y_a_shot)
# mae_t2_a_shot=mean_absolute_error(p_t2_a_shot,v2_y_a_shot)
# # mae_t3_a_shot=mean_absolute_error(p_t3_a_shot,v3_y_a_shot)

# mae_t1_a_cross=mean_absolute_error(p_t1_a_cross,v1_y_a_cross)
# mae_t2_a_cross=mean_absolute_error(p_t2_a_cross,v2_y_a_cross)
# # mae_t3_a_cross=mean_absolute_error(p_t3_a_cross,v3_y_a_cross)

# #%% note
# # MAE=mean_absolute_error(test,v1_y_h_shoton)
# # MSE=mean_squared_error(test,v1_y_h_shoton)
# # RMSE=math.sqrt(MSE)
# #%% Grid Search
# model = ElasticNet()
# cv = [(slice(None), slice(None))]
# grid = dict()
# grid['alpha'] = arange(0, 1, 0.01)
# grid['l1_ratio'] = arange(0, 1, 0.01)
# search = GridSearchCV(model, grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
# results = search.fit(t2_x, t2_y_h_cross)
# # summarize
# print('MSE: %.3f' % results.best_score_)
# print('Config: %s' % results.best_params_)

