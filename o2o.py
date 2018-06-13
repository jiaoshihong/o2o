import os, sys, pickle

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from datetime import date
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler


weekdaycols = ['weekday_' + str(i) for i in range(1,8)]

dfoff = pd.read_csv('../datalab/dfoff.csv', 
                        dtype = {
                            'Date_received' : np.object, 
                            'Date' : np.object,
                            'Coupon_id' : np.object,
                            'Discount_rate' : np.object,
                            'Distance' : np.object
                        }
                    )
dfoff['Date_received'] = dfoff['Date_received'].replace(np.nan, 'null').astype(object)
dfoff['Date'] = dfoff['Date'].replace(np.nan, 'null').astype(object)
dfoff['Coupon_id'] = dfoff['Coupon_id'].replace(np.nan, 'null').astype(object)
dfoff['Discount_rate'] = dfoff['Discount_rate'].replace(np.nan, 'null').astype(object)
dfoff['Distance'] = dfoff['Distance'].replace(np.nan, 'null').astype(object)

train = pd.read_csv('../datalab/train.csv'#, 
                        # dtype = {
                        #     'Date_received' : np.object, 
                        #     'Date' : np.object,
                        #     'Coupon_id' : np.object,
                        #     'Discount_rate' : np.object,
                        #     'Distance' : np.object
                        # }
                    )
valid = pd.read_csv('../datalab/valid.csv'#, 
                        # dtype = {
                        #     'Date_received' : np.object, 
                        #     'Date' : np.object,
                        #     'Coupon_id' : np.object,
                        #     'Discount_rate' : np.object,
                        #     'Distance' : np.object
                        # }
                    )  

dftest = pd.read_csv('../datalab/dftest.csv'#, 
                        # dtype = {
                        #     'Discount_rate' : np.object,
                        #     'Distance' : np.object
                        # }
                    )  

# feature
original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian','distance', 'weekday', 'weekday_type'] + weekdaycols
print(len(original_feature),original_feature)


predictors = original_feature
print("predictors", predictors)

def check_model(data, predictors):
    
    classifier = lambda: SGDClassifier(
        loss='log', 
        penalty='elasticnet', 
        fit_intercept=True, 
        max_iter=100, 
        shuffle=True, 
        n_jobs=1,
        class_weight=None)

    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('en', classifier())
    ])

    parameters = {
        'en__alpha': [ 0.001, 0.01, 0.1],
        'en__l1_ratio': [ 0.001, 0.01, 0.1]
    }

    folder = StratifiedKFold(n_splits=3, shuffle=True)
    
    grid_search = GridSearchCV(
        model, 
        parameters, 
        cv=folder, 
        n_jobs=1, # avoid freeze https://medium.com/@zector1030/sklearn-%E5%B0%8D-svc-%E4%BD%9C-gridsearchcv-%E5%B0%8E%E8%87%B4-freeze-8b966001161d
        verbose=1)
    # data[predictors]=np.ascontiguousarray(data[predictors])
    grid_search = grid_search.fit(data[predictors], 
                                  data['label'])
    
    return grid_search

print(train.head(2))

if not os.path.isfile('../datalab/1_model.pkl'):
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('../datalab/1_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('../datalab/1_model.pkl', 'rb') as f:
        model = pickle.load(f)

# valid predict
y_valid_pred = model.predict_proba(valid[predictors])
valid1 = valid.copy()
valid1['pred_prob'] = y_valid_pred[:, 1]
valid1.head(2)

# avgAUC calculation
vg = valid1.groupby(['Coupon_id'])
aucs = []
for i in vg:
    tmpdf = i[1] 
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))   

# test prediction for submission
y_test_pred = model.predict_proba(dftest[predictors])
dftest1 = dftest[['User_id','Coupon_id','Date_received']].copy()
dftest1['label'] = y_test_pred[:,1]
dftest1.to_csv('submit1.csv', index=False, header=False)
dftest1.head()

feature = dfoff[(dfoff['Date'] < '20160516') | ((dfoff['Date'] == 'null') & (dfoff['Date_received'] < '20160516'))].copy()
data = dfoff[(dfoff['Date_received'] >= '20160516') & (dfoff['Date_received'] <= '20160615')].copy()
print(data['label'].value_counts())

fdf = feature.copy()

# key of user
u = fdf[['User_id']].copy().drop_duplicates()

# u_coupon_count : num of coupon received by user
u1 = fdf[fdf['Date_received'] != 'null'][['User_id']].copy()
u1['u_coupon_count'] = 1
u1 = u1.groupby(['User_id'], as_index = False).count()
u1.head(2)

# u_buy_count : times of user buy offline (with or without coupon)
u2 = fdf[fdf['Date'] != 'null'][['User_id']].copy()
u2['u_buy_count'] = 1
u2 = u2.groupby(['User_id'], as_index = False).count()
u2.head(2)

# u_buy_with_coupon : times of user buy offline (with coupon)
u3 = fdf[((fdf['Date'] != 'null') & (fdf['Date_received'] != 'null'))][['User_id']].copy()
u3['u_buy_with_coupon'] = 1
u3 = u3.groupby(['User_id'], as_index = False).count()
u3.head(2)

# u_merchant_count : num of merchant user bought from
u4 = fdf[fdf['Date'] != 'null'][['User_id', 'Merchant_id']].copy()
u4.drop_duplicates(inplace = True)
u4 = u4.groupby(['User_id'], as_index = False).count()
u4.rename(columns = {'Merchant_id':'u_merchant_count'}, inplace = True)
u4.head(2)

# u_min_distance
utmp = fdf[(fdf['Date'] != 'null') & (fdf['Date_received'] != 'null')][['User_id', 'distance']].copy()
utmp.replace(-1, np.nan, inplace = True)
u5 = utmp.groupby(['User_id'], as_index = False).min()
u5.rename(columns = {'distance':'u_min_distance'}, inplace = True)
u6 = utmp.groupby(['User_id'], as_index = False).max()
u6.rename(columns = {'distance':'u_max_distance'}, inplace = True)
u7 = utmp.groupby(['User_id'], as_index = False).mean()
u7.rename(columns = {'distance':'u_mean_distance'}, inplace = True)
u8 = utmp.groupby(['User_id'], as_index = False).median()
u8.rename(columns = {'distance':'u_median_distance'}, inplace = True)
u8.head(2)

u.shape, u1.shape, u2.shape, u3.shape, u4.shape, u5.shape, u6.shape, u7.shape, u8.shape

# merge all the features on key User_id
user_feature = pd.merge(u, u1, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u2, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u3, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u4, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u5, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u6, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u7, on = 'User_id', how = 'left')
user_feature = pd.merge(user_feature, u8, on = 'User_id', how = 'left')


# calculate rate

user_feature['u_use_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float')/user_feature['u_coupon_count'].astype('float')
user_feature['u_buy_with_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float')/user_feature['u_buy_count'].astype('float')
user_feature = user_feature.fillna(0)
user_feature.head(2)

# add user feature to data on key User_id
data2 = pd.merge(data, user_feature, on = 'User_id', how = 'left').fillna(0)

# split data2 into valid and train
train, valid = train_test_split(data2, test_size = 0.2, stratify = data2['label'], random_state=100)

# model2
predictors = original_feature + user_feature.columns.tolist()[1:]
print(len(predictors), predictors)

if not os.path.isfile('../datalab/2_model.pkl'):
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('../datalab/2_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('../datalab/2_model.pkl', 'rb') as f:
        model = pickle.load(f)

# valid set performance 
valid['pred_prob'] = model.predict_proba(valid[predictors])[:,1]
validgroup = valid.groupby(['Coupon_id'])
aucs = []
for i in validgroup:
    tmpdf = i[1] 
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    
    aucs.append(auc(fpr, tpr))
    #aucs.append(roc_auc_score(tmpdf['label'], tmpdf['pred_prob']))
print(np.average(aucs))     


# key of merchant
m = fdf[['Merchant_id']].copy().drop_duplicates()

# m_coupon_count : num of coupon from merchant
m1 = fdf[fdf['Date_received'] != 'null'][['Merchant_id']].copy()
m1['m_coupon_count'] = 1
m1 = m1.groupby(['Merchant_id'], as_index = False).count()
m1.head(2)

# m_sale_count : num of sale from merchant (with or without coupon)
m2 = fdf[fdf['Date'] != 'null'][['Merchant_id']].copy()
m2['m_sale_count'] = 1
m2 = m2.groupby(['Merchant_id'], as_index = False).count()
m2.head(2)

# m_sale_with_coupon : num of sale from merchant with coupon usage
m3 = fdf[(fdf['Date'] != 'null') & (fdf['Date_received'] != 'null')][['Merchant_id']].copy()
m3['m_sale_with_coupon'] = 1
m3 = m3.groupby(['Merchant_id'], as_index = False).count()
m3.head(2)

# m_min_distance
mtmp = fdf[(fdf['Date'] != 'null') & (fdf['Date_received'] != 'null')][['Merchant_id', 'distance']].copy()
mtmp.replace(-1, np.nan, inplace = True)
m4 = mtmp.groupby(['Merchant_id'], as_index = False).min()
m4.rename(columns = {'distance':'m_min_distance'}, inplace = True)
m5 = mtmp.groupby(['Merchant_id'], as_index = False).max()
m5.rename(columns = {'distance':'m_max_distance'}, inplace = True)
m6 = mtmp.groupby(['Merchant_id'], as_index = False).mean()
m6.rename(columns = {'distance':'m_mean_distance'}, inplace = True)
m7 = mtmp.groupby(['Merchant_id'], as_index = False).median()
m7.rename(columns = {'distance':'m_median_distance'}, inplace = True)
m7.head(2)

m.shape, m1.shape, m2.shape, m3.shape, m4.shape, m5.shape, m6.shape, m7.shape

merchant_feature = pd.merge(m, m1, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m2, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m3, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m4, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m5, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m6, on = 'Merchant_id', how = 'left')
merchant_feature = pd.merge(merchant_feature, m7, on = 'Merchant_id', how = 'left')
merchant_feature = merchant_feature.fillna(0)
merchant_feature.head(5)

merchant_feature['m_coupon_use_rate'] = merchant_feature['m_sale_with_coupon'].astype('float')/merchant_feature['m_coupon_count'].astype('float')
merchant_feature['m_sale_with_coupon_rate'] = merchant_feature['m_sale_with_coupon'].astype('float')/merchant_feature['m_sale_count'].astype('float')
merchant_feature = merchant_feature.fillna(0)
merchant_feature.head(2)

# add merchant feature to data2
data3 = pd.merge(data2, merchant_feature, on = 'Merchant_id', how = 'left').fillna(0)

# split data3 into train/valid
train, valid = train_test_split(data3, test_size = 0.2, stratify = data3['label'], random_state=100)

# model 3
predictors = original_feature + user_feature.columns.tolist()[1:] + merchant_feature.columns.tolist()[1:]
print(predictors)

if not os.path.isfile('../datalab/3_model.pkl'):
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('../datalab/3_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('../datalab/3_model.pkl', 'rb') as f:
        model = pickle.load(f)

valid['pred_prob'] = model.predict_proba(valid[predictors])[:,1]
validgroup = valid.groupby(['Coupon_id'])
aucs = []
for i in validgroup:
    tmpdf = i[1] 
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))

print("please enter anykey for end")
input()
