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

train = pd.read_csv('../datalab/train.csv'#, 
                        # dtype = {
                        #     'Date_received' : np.object, 
                        #     'Date' : np.object,
                        #     'Coupon_id' : np.object,
                        #     'Discount_rate' : np.object,
                        #     'Distance' : np.object
                        # }
                    )  

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

vg = dftest1.groupby(['Coupon_id'])
aucs = []
for i in vg:
    tmpdf = i[1] 
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))

print("please enter anykey for end")
input()
