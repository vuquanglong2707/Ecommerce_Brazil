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

######### Phân khúc khách hàng với phân tích RFM
#Phân tích theo 3 đặc điểm : Recency (Số lần truy cập),Frequency(Tần suất), Monetary (Số tiền đã chi)
## getting order id by customer purchases 
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

## the number of times a unique customer has made purchase
frequency=pd.DataFrame(df_customer_order_rev.groupby('customer_unique_id')['customer_id'].count())
monetary=pd.DataFrame(df_customer_order_rev[['customer_unique_id','payment_value']].groupby('customer_unique_id')['payment_value'].sum())

# the receny of visit, total monetary spent and freqency of purchase by each customer is found out and merged
df_rfm=pd.merge(recency,frequency,on='customer_unique_id')
df_rfm=pd.merge(df_rfm,monetary,on='customer_unique_id')

## Freqency - Number of purchase made
## Recency- Days from last purchase
## Monetary-- total amount purchase for by a customer
df_rfm.drop(['order_purchase_timestamp'],axis=1,inplace=True)
df_rfm.reset_index(inplace=True)
df_rfm.columns=['Cust_unique_Id','Recency','Frequency','Monetary']

#use CustomerID as index
df_rfm.set_index('Cust_unique_Id',inplace=True)

#print(df_rfm)
df_rfm

## the descriptive stats for the RFM analysis
df_rfm.describe()
(df_rfm[df_rfm['Frequency']>1].shape[0]/96095)*100

# Plot RFM distributions
plt.figure(figsize=(12,10))

# Plot distribution of R
plt.subplot(3, 1, 1); sns.distplot(df_rfm['Recency'],kde=False)

# Plot distribution of F
plt.subplot(3, 1, 2); sns.distplot(df_rfm['Frequency'],kde=False)

# Plot distribution of M
plt.subplot(3, 1, 3); sns.distplot(df_rfm['Monetary'],kde=False)

# Show the plot
plt.show()
quantiles = df_rfm.quantile(q=[0.25,0.5,0.75])
quantiles.to_dict()

# Arguments (x = value, p = recency, monetary_value, frequency, d = quartiles dict)
def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

#create rfm segmentation table
rfm_segmentation = df_rfm
rfm_segmentation['R_Quartile'] = rfm_segmentation['Recency'].apply(RScore, args=('Recency',quantiles,))
rfm_segmentation['F_Quartile'] = rfm_segmentation['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
rfm_segmentation['M_Quartile'] = rfm_segmentation['Monetary'].apply(FMScore, args=('Monetary',quantiles,))
rfm_segmentation['RFMScore'] = rfm_segmentation.R_Quartile.map(str) \
                            + rfm_segmentation.F_Quartile.map(str) \
                            + rfm_segmentation.M_Quartile.map(str)
rfm_segmentation.head()
print(rfm_segmentation)

### how many customers are in each segment
print("Best Customers: ",len(rfm_segmentation[rfm_segmentation['RFMScore']=='444']))
print('Loyal Customers: ',len(rfm_segmentation[rfm_segmentation['F_Quartile']==4]))
print("Big Spenders: ",len(rfm_segmentation[rfm_segmentation['M_Quartile']==4]))
print('Almost Lost: ', len(rfm_segmentation[rfm_segmentation['RFMScore']=='244']))
print('Lost Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='144']))
print('Lost Cheap Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='111']))
best_cust=rfm_segmentation[rfm_segmentation['RFMScore']=='444']

#calculate and show correlations
corr_matrix = best_cust.corr()
sns.heatmap(corr_matrix)

##merging sellers with orderid sold
df_seller_seg=pd.merge(df_items[['order_id','seller_id','price']],df_orders[['order_id','order_purchase_timestamp','order_delivered_carrier_date']],on='order_id')

##merging review scores of the order with the order ids
df_seller_seg=pd.merge(df_seller_seg,df_reviews[['order_id','review_score']],on='order_id')

##converting dates to datetime format
df_seller_seg['order_purchase_timestamp']=pd.to_datetime(df_seller_seg['order_purchase_timestamp']).dt.date
df_seller_seg['order_delivered_carrier_date']=pd.to_datetime(df_seller_seg['order_delivered_carrier_date']).dt.date
df_seller_seg['days_to_del']=(df_seller_seg['order_delivered_carrier_date']-df_seller_seg['order_purchase_timestamp']).dt.days

## dropping unwanted columns
df_seller_seg.drop(['order_purchase_timestamp','order_delivered_carrier_date'],axis=1,inplace=True)

## filling missing values with mean values
df_seller_seg['days_to_del'].fillna(df_seller_seg['days_to_del'].mean(),inplace=True)

## removing negative dates, these are outliers
df_seller_seg_1=df_seller_seg[df_seller_seg['days_to_del']>=0]

## makeing total number of sales, amount of total sales, average review, avergage days to deliver to carrier
df_seller_seg_2=df_seller_seg_1.groupby('seller_id').agg({'order_id':'count','price':'sum','review_score':'mean','days_to_del':'mean'})
df_seller_seg_2.columns=['Tot_sales','Tot_amount','Avg_review','Avg_delivery']

## filling missing values with mean values
df_seller_seg_2['Avg_delivery'].fillna(df_seller_seg_2['Avg_delivery'].mean(),inplace=True)

##splitting into quartiles
quantiles = df_seller_seg_2.quantile(q=[0.25,0.5,0.75])
quantiles.to_dict()
print('seller seg 2 : \n', df_seller_seg_2)
print('quantiles: \n',quantiles)

# feature statistic distribution
df_seller_seg_2.describe()
df_seller_seg_2['Tot_sales_Quartile'] = df_seller_seg_2['Tot_sales'].apply(FMScore, args=('Tot_sales',quantiles,))
df_seller_seg_2['Avg_review_Quartile'] = df_seller_seg_2['Avg_review'].apply(FMScore, args=('Avg_review',quantiles,))
df_seller_seg_2['Tot_amount_Quartile'] = df_seller_seg_2['Tot_amount'].apply(FMScore, args=('Tot_amount',quantiles,))
df_seller_seg_2['Avg_delivery_Quartile'] = df_seller_seg_2['Avg_delivery'].apply(RScore, args=('Avg_delivery',quantiles,))
df_seller_seg_2['SellerScore'] = df_seller_seg_2['Tot_sales_Quartile'].map(str) \
                            + df_seller_seg_2['Tot_amount_Quartile'].map(str) \
                            + df_seller_seg_2['Avg_review_Quartile'] .map(str) \
                            + df_seller_seg_2['Avg_delivery_Quartile'].map(str)
df_seller_seg_2.head()
print('df_seller_2 : \n',df_seller_seg_2 )

## how many sellers are in each segment
### how many sellers are in each segment
print("Best Sellers: ",len(df_seller_seg_2[df_seller_seg_2['SellerScore']=='4444']))
print('highest Reviewed Sellers: ',len(df_seller_seg_2[df_seller_seg_2['Avg_review_Quartile']==4]))
print("Fastest delivering sellers: ",len(df_seller_seg_2[df_seller_seg_2['Avg_delivery_Quartile']==4]))

## there can be overlap in this classification . For example highest reviewed sellers can be best Fastest delivering sellers
Fastest_delivering =df_seller_seg_2[df_seller_seg_2['Avg_delivery_Quartile']==4].reset_index()
highest_Reviewed=df_seller_seg_2[df_seller_seg_2['Avg_review_Quartile']==4].reset_index()
len(set(list(Fastest_delivering['seller_id'])).intersection(highest_Reviewed['seller_id'].tolist()))
best_cust=df_seller_seg_2[df_seller_seg_2['SellerScore']=='4444']

#calculate and show correlations
corr_matrix = best_cust.corr()
sns.heatmap(corr_matrix)
df_customer_order_rev=pd.merge(df_customer_order,paid,on='order_id')

## the count, mean, standard deviation,minimum, maximum,qunatiles are displayed of the total spend by customers
pd.DataFrame(df_customer_order_rev.groupby('customer_unique_id')['payment_value'].sum().describe())
pd.DataFrame(df_customer_order_rev.groupby('customer_unique_id')['customer_id'].count().describe())
pd.DataFrame(df_customer_order_rev.groupby('customer_city')['customer_id'].count().describe())
df_state=pd.DataFrame(df_customer_order_rev.groupby('customer_state')['customer_id'].count()).reset_index()
plt.figure(figsize=(12,8))
plt.bar(df_state['customer_state'],df_state['customer_id'])
plt.show()
df_state['customer_id'].describe()
df_sellers_stat=pd.merge(df_sellers,df_items,on='seller_id')

## dropping unwanted columns
df_sellers_stat.drop(['seller_zip_code_prefix','order_item_id','product_id'],axis=1,inplace=True)

## the count, mean, standard deviation,minimum, maximum,qunatiles are displayed of the total spend by customers
pd.DataFrame(df_sellers_stat.groupby('seller_id')['price'].sum().describe())
pd.DataFrame(df_sellers_stat.groupby('seller_id')['order_id'].count().describe())
pd.DataFrame(df_sellers_stat.groupby('seller_city')['order_id'].count().describe())
df_state=pd.DataFrame(df_sellers_stat.groupby('seller_state')['order_id'].count()).reset_index()
plt.figure(figsize=(12,8))
plt.bar(df_state['seller_state'],df_state['order_id'])
plt.show()
df_state['order_id'].describe()

## combine orders , reviews, payments dataset with customer dataset and dropping unwanted columns
df1=pd.merge(df_customer.drop(columns=['customer_zip_code_prefix']),df_orders[['customer_id','order_id','order_purchase_timestamp']],on='customer_id')
df2=pd.merge(df1,df_reviews[['order_id','review_score']],on='order_id')
paid=df_payments[['order_id','payment_value']].groupby('order_id').sum().reset_index()
df3=pd.merge(df2,paid,on='order_id')

## making purchase date in datetime format
df3['order_purchase_timestamp']=pd.to_datetime(df3['order_purchase_timestamp']).dt.date
number_of_days_for_purchase=180
max_date_in_data= df3['order_purchase_timestamp'].max()
data_split_date=max_date_in_data-dt.timedelta(days=number_of_days_for_purchase)
df_full=df3[df3['order_purchase_timestamp']<=data_split_date]
df_last=df3[df3['order_purchase_timestamp']>data_split_date]
df_last_180=pd.DataFrame({'customer_unique_id':df3['customer_unique_id'].values.tolist()})
df_last_180=df_last_180.merge(df_last.groupby(['customer_unique_id'])['payment_value'].sum().reset_index(),how='outer',on='customer_unique_id')
df_last_180.fillna(0,inplace=True)
df_last_180['purchased']=np.where(df_last_180['payment_value']>0, 1,0)
df_last_180.head()

## total amount per customer
tot_Amount=df_full.groupby('customer_unique_id')['payment_value'].sum().reset_index().rename(columns={'payment_value':'total_amount'})

## average review given
avg_review=df_full.groupby('customer_unique_id')['review_score'].mean().reset_index().rename(columns={'review_score':'avg_review'})

## months between first purchase and today
min_max_date=df_full.groupby('customer_unique_id')['order_purchase_timestamp'].agg([min,max])
min_max_date['diff_first_today']=(dt.datetime.today().date()-min_max_date['min']).dt.days

## months from first to last purchase
min_max_date['max']=pd.to_datetime(min_max_date['max'])
min_max_date['min']=pd.to_datetime(min_max_date['min'])
min_max_date['diff_first_last']=(min_max_date['max']-min_max_date['min']).dt.days

## recency of Sales 
max_date=df_full['order_purchase_timestamp'].max()
min_max_date['recency']=(np.datetime64(max_date)-min_max_date['max'])/np.timedelta64(1, 'M')

## Frequency of Sales
frequency=df_full.groupby('customer_unique_id')['order_id'].count().reset_index().rename(columns={'order_id':'frequency'})

## joining all the engineered features
dataset=pd.merge(tot_Amount,avg_review,on='customer_unique_id')
dataset=pd.merge(dataset,min_max_date,on='customer_unique_id')
dataset=pd.merge(dataset,frequency,on='customer_unique_id')
dataset=pd.merge(dataset,df_full[['customer_unique_id','customer_city','customer_state']],on='customer_unique_id')
dataset.drop(['min','max'],axis=1,inplace=True)

### label encoding city and state names
encoder=LabelEncoder()
dataset['customer_city']=encoder.fit_transform(dataset['customer_city'])
dataset['customer_state']=encoder.fit_transform(dataset['customer_state'])

##merging with the label dataset we have created 
dataset_full=dataset.merge(df_last_180[['customer_unique_id','purchased']],on='customer_unique_id')
dataset_full.drop(columns='customer_unique_id',inplace=True)

##splitting to train and test dataset
X_train,X_test,y_train,y_test=train_test_split(dataset_full.iloc[:,:-1],dataset_full.iloc[:,-1], test_size=0.2, random_state=31)

## calculating gini scores for the models
print('Xtrain= \n',X_train)
X_train.to_csv(r'xtrain.csv', index = False)
print('XTest= \n',X_test)
X_test.to_csv(r'xtest.csv', index = False)
print('ytrain= \n',y_train)
y_train.to_csv(r'ytrain.csv', index = False)
print('Ytest=\n',y_test)
y_test.to_csv(r'ytest.csv', index = False)

