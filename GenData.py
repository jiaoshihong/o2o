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

# import xgboost as xgb
# import lightgbm as lgb

# display for this notebook
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

def GenTrain_ValidData():

    dfoff = pd.read_csv('datalab/ccf_offline_stage1_train.csv', 
                            dtype = {
                                'Date_received' : np.object, 
                                'Date' : np.object,
                                'Coupon_id' : np.object,
                                'Discount_rate' : np.object,
                                'Distance' : np.object
                            }
                        )               
    dftest = pd.read_csv('datalab/ccf_offline_stage1_test_revised.csv',
                            dtype = {
                                'Date_received' : np.object, 
                                'Coupon_id' : np.object,
                                'Discount_rate' : np.object,
                                'Distance' : np.object
                            }
                        )

    dfon = pd.read_csv('datalab/ccf_online_stage1_train.csv',
                            dtype = {
                                'Date_received' : np.object, 
                                'Date' : np.object,
                                'Coupon_id' : np.object,
                                'Discount_rate' : np.object
                            }
                        )

    print(dfoff.head(2))
    # dfoff.info()

    # dfoff['Date_received'] = dfoff['Date_received'].astype(object)
    dfoff['Date_received'] = dfoff['Date_received'].replace(np.nan, 'null').astype(object)
    # dfoff['Date'] = dfoff['Date'].astype('object')
    dfoff['Date'] = dfoff['Date'].replace(np.nan, 'null').astype(object)
    # dfoff['Coupon_id'] = dfoff['Coupon_id'].astype('object')
    dfoff['Coupon_id'] = dfoff['Coupon_id'].replace(np.nan, 'null').astype(object)
    # dfoff['Discount_rate'] = dfoff['Discount_rate'].astype('object')
    dfoff['Discount_rate'] = dfoff['Discount_rate'].replace(np.nan, 'null').astype(object)
    # dfoff['Distance'] = dfoff['Distance'].astype('object')
    dfoff['Distance'] = dfoff['Distance'].replace(np.nan, 'null').astype(object)

    # dftest['Date_received'] = dftest['Date_received'].astype('object')
    dftest['Date_received'] = dftest['Date_received'].replace(np.nan, 'null').astype(object)
    # dftest['Coupon_id'] = dftest['Coupon_id'].astype('object')
    dftest['Coupon_id'] = dftest['Coupon_id'].replace(np.nan, 'null').astype(object)
    # dftest['Discount_rate'] = dftest['Discount_rate'].astype('object')
    dftest['Discount_rate'] = dftest['Discount_rate'].replace(np.nan, 'null').astype(object)
    # dftest['Distance'] = dftest['Distance'].astype('object')
    dftest['Distance'] = dftest['Distance'].replace(np.nan, 'null').astype(object)

    # dfon['Date_received'] = dfon['Date_received'].astype('object')
    dfon['Date_received'] = dfon['Date_received'].replace(np.nan, 'null').astype(object)
    # dfon['Date'] = dfon['Date'].astype('object')
    dfon['Date'] = dfon['Date'].replace(np.nan, 'null').astype(object)
    # dfon['Coupon_id'] = dfon['Coupon_id'].astype('object')
    dfon['Coupon_id'] = dfon['Coupon_id'].replace(np.nan, 'null').astype(object)
    # dfon['Discount_rate'] = dfon['Discount_rate'].astype('object')
    dfon['Discount_rate'] = dfon['Discount_rate'].replace(np.nan, 'null').astype(object)

    print(dfoff.head(5))

    dfoff.info()
    dfon.info()
    dftest.info()

    print('有优惠券，购买商品条数', dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] != 'null')].shape[0])
    print('无优惠券，购买商品条数', dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] != 'null')].shape[0])
    print('有优惠券，不购买商品条数', dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] == 'null')].shape[0])
    print('无优惠券，不购买商品条数', dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] == 'null')].shape[0])
    print('1. User_id in training set but not in test set', set(dftest['User_id']) - set(dfoff['User_id']))
    print('2. Merchant_id in training set but not in test set', set(dftest['Merchant_id']) - set(dfoff['Merchant_id']))

    print('Discount_rate 类型:',dfoff['Discount_rate'].unique())
    print('Distance 类型:', dfoff['Distance'].unique())

    def getDiscountType(row):
        if row == 'null':
            return 'null'
        elif ':' in row:
            return 1
        else:
            return 0

    def convertRate(row):
        """Convert discount to rate"""
        if row == 'null':
            return 1.0
        elif ':' in row:
            rows = row.split(':')
            return 1.0 - float(rows[1])/float(rows[0])
        else:
            return float(row)

    def getDiscountMan(row):
        if ':' in row:
            rows = row.split(':')
            return int(rows[0])
        else:
            return 0

    def getDiscountJian(row):
        if ':' in row:
            rows = row.split(':')
            return int(rows[1])
        else:
            return 0

    def processData(df):
        
        # convert discunt_rate
        df['discount_rate'] = df['Discount_rate'].apply(convertRate)
        df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
        df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
        df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
        print(df['discount_rate'].unique())
        
        # convert distance
        df['distance'] = df['Distance'].replace('null', -1).astype(int)
        print(df['distance'].unique())
        return df

    dfoff = processData(dfoff)
    dftest = processData(dftest)

    date_received = dfoff['Date_received'].unique()
    date_received = sorted(date_received[date_received != 'null'])

    date_buy = dfoff['Date'].unique()
    date_buy = sorted(date_buy[date_buy != 'null'])

    date_buy = sorted(dfoff[dfoff['Date'] != 'null']['Date'])
    print('优惠券收到日期从',date_received[0],'到', date_received[-1])
    print('消费日期从', date_buy[0], '到', date_buy[-1])

    couponbydate = dfoff[dfoff['Date_received'] != 'null'][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
    couponbydate.columns = ['Date_received','count']

    buybydate = dfoff[(dfoff['Date'] != 'null') & (dfoff['Date_received'] != 'null')][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
    buybydate.columns = ['Date_received','count']

    # sns.set_style('ticks')
    # sns.set_context("notebook", font_scale= 1.4)
    # plt.figure(figsize = (12,8))
    # date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')

    # plt.subplot(211)
    # plt.bar(date_received_dt, couponbydate['count'], label = 'number of coupon received' )
    # plt.bar(date_received_dt, buybydate['count'], label = 'number of coupon used')
    # plt.yscale('log')
    # plt.ylabel('Count')
    # plt.legend()

    # plt.subplot(212)
    # plt.bar(date_received_dt, buybydate['count']/couponbydate['count'])
    # plt.ylabel('Ratio(coupon used/coupon received)')
    # plt.tight_layout()

    # plt.show()

    def getWeekday(row):
        if row == 'null':
            return row
        else:
            return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

    dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
    dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

    # weekday_type :  周六和周日为1，其他为0
    dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
    dftest['weekday_type'] = dftest['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )

    # change weekday to one-hot encoding 
    weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
    print(weekdaycols)

    tmpdf = pd.get_dummies(dfoff['weekday'].replace('null', np.nan))
    tmpdf.columns = weekdaycols
    dfoff[weekdaycols] = tmpdf

    tmpdf = pd.get_dummies(dftest['weekday'].replace('null', np.nan))
    tmpdf.columns = weekdaycols
    dftest[weekdaycols] = tmpdf

    def label(row):
        if row['Date_received'] == 'null':
            return -1
        if row['Date'] != 'null':
            td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
            if td <= pd.Timedelta(15, 'D'):
                return 1
        return 0
    dfoff['label'] = dfoff.apply(label, axis = 1)

    print(dfoff['label'].value_counts())

    print('已有columns：',dfoff.columns.tolist())

    dfoff.head(2)
    dfoff.to_csv('temp1.csv')

    # data split
    df = dfoff[dfoff['label'] != -1].copy()
    print("dfinfo", df.info())
    train = df[(df['Date_received'] < '20160516')].copy()
    valid = df[(df['Date_received'] >= '20160516') & (df['Date_received'] <= '20160615')].copy()
    print(train['label'].value_counts())
    print(valid['label'].value_counts())