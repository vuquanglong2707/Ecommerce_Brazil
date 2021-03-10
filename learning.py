# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 08:45:36 2021

@author: Quang Long
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "data/" directory.

import time, warnings
import datetime as dt

#visualizations
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
# %matplotlib inline
import seaborn as sns

# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import r2_score,accuracy_score,classification_report
# Splitting data into training/testing
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans

# Importing libraries for building the neural network
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import xgboost as xgb
from eli5.sklearn import PermutationImportance
import eli5

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

##Đọc dữ liệu từ file
df_customer=pd.read_csv('olist_customers_dataset.csv')
df_location=pd.read_csv('olist_geolocation_dataset.csv')
df_items=pd.read_csv('olist_order_items_dataset.csv')
df_payments=pd.read_csv('olist_order_payments_dataset.csv')
df_reviews=pd.read_csv('olist_order_reviews_dataset.csv')
df_products=pd.read_csv('olist_products_dataset.csv')
df_orders=pd.read_csv('olist_orders_dataset.csv')
df_sellers=pd.read_csv('olist_sellers_dataset.csv')
df_name_trans=pd.read_csv('product_category_name_translation.csv')
df_customer_order=pd.merge(df_customer,df_orders[['order_id','customer_id','order_purchase_timestamp']],on='customer_id')

## payments in same order id are combined to get total spending on an order
paid=df_payments[['order_id','payment_value']].groupby('order_id').sum().reset_index()
## now the total payment by each order is merged to the cutomer who has bought it to find the total amount purchase
df_customer_order_rev=pd.merge(df_customer_order,paid,on='order_id')

## unwanted columns are dropped
df_customer_order_rev.drop(['customer_zip_code_prefix','customer_city','customer_state'],axis=1,inplace=True)
df_customer_order_rev['order_purchase_timestamp']=pd.to_datetime(df_customer_order_rev['order_purchase_timestamp']).dt.date

## find the last date on which customer made the purchase
recency=pd.DataFrame(df_customer_order_rev.groupby('customer_unique_id')['order_purchase_timestamp'].max())

## we take the maximum date of purchase made by customers as the date to calculate the recency of the purchase
## 2018-10-17
recency['recent_days']=recency['order_purchase_timestamp'].max()-recency['order_purchase_timestamp']
recency['recent_days']=recency['recent_days'].dt.days