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

############################## Phân khúc khách hàng với phân tích RFM  ###################
#Phân tích theo 3 đặc điểm : Recency (Số lần truy cập),Frequency(Tần suất), Monetary (Số tiền đã chi)
## Lọc order_id khi khách hàng mua hàng
df_customer_order=pd.merge(df_customer,df_orders[['order_id','customer_id','order_purchase_timestamp']],on='customer_id')
print(df_customer_order)
## tông tiền hóa đơn theo order_id
paid=df_payments[['order_id','payment_value']].groupby('order_id').sum().reset_index()

## merge customer với tổng tiền hóa đơn
df_customer_order_rev=pd.merge(df_customer_order,paid,on='order_id')

## Lọc các cột không cần thiết
df_customer_order_rev.drop(['customer_zip_code_prefix','customer_city','customer_state'],axis=1,inplace=True)
df_customer_order_rev['order_purchase_timestamp']=pd.to_datetime(df_customer_order_rev['order_purchase_timestamp']).dt.date

## Tìm ngày tối đã mà khách hàng mua hàng
recency=pd.DataFrame(df_customer_order_rev.groupby('customer_unique_id')['order_purchase_timestamp'].max())

## Lấy ngày tối đa để tính ngày mua hàng gần nhất
recency['recent_days']=recency['order_purchase_timestamp'].max()-recency['order_purchase_timestamp']
recency['recent_days']=recency['recent_days'].dt.days

## tần suất mua hàng của khách hàng
frequency=pd.DataFrame(df_customer_order_rev.groupby('customer_unique_id')['customer_id'].count())
monetary=pd.DataFrame(df_customer_order_rev[['customer_unique_id','payment_value']].groupby('customer_unique_id')['payment_value'].sum())

# Kết hợp lần gần đây nhất, tổng tiền đã mua và tần suất mua hàng của khách hàng
df_rfm=pd.merge(recency,frequency,on='customer_unique_id')
df_rfm=pd.merge(df_rfm,monetary,on='customer_unique_id')

## Freqency - tần suất mua hàng
## Recency- Số ngày kể từ lần mua hàng cuối cùng
## Monetary-- Tổng số tiền mua hàng của khách hàng
df_rfm.drop(['order_purchase_timestamp'],axis=1,inplace=True)
df_rfm.reset_index(inplace=True)
df_rfm.columns=['Cust_unique_Id','Recency','Frequency','Monetary']

#Sử dụng customerID để là chỉ số
df_rfm.set_index('Cust_unique_Id',inplace=True)

#print(df_rfm)
df_rfm

## thống kê mô tả cho phân tích RFM
df_rfm.describe()
## Xác định số khách hàng chỉ mua 1 lần và khách hàng quay lại 
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

#RFM quantities
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

#Tạo bảng phân tích RFM
rfm_segmentation = df_rfm
rfm_segmentation['R_Quartile'] = rfm_segmentation['Recency'].apply(RScore, args=('Recency',quantiles,))
rfm_segmentation['F_Quartile'] = rfm_segmentation['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
rfm_segmentation['M_Quartile'] = rfm_segmentation['Monetary'].apply(FMScore, args=('Monetary',quantiles,))
rfm_segmentation['RFMScore'] = rfm_segmentation.R_Quartile.map(str) \
                            + rfm_segmentation.F_Quartile.map(str) \
                            + rfm_segmentation.M_Quartile.map(str)
rfm_segmentation.head()
print(rfm_segmentation)

### Số khách hàng trong mỗi phân khúc là 
print("Best Customers: ",len(rfm_segmentation[rfm_segmentation['RFMScore']=='444']))
print('Loyal Customers: ',len(rfm_segmentation[rfm_segmentation['F_Quartile']==4]))
print("Big Spenders: ",len(rfm_segmentation[rfm_segmentation['M_Quartile']==4]))
print('Almost Lost: ', len(rfm_segmentation[rfm_segmentation['RFMScore']=='244']))
print('Lost Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='144']))
print('Lost Cheap Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='111']))


########################## Đánh giá người bán   ##############################
#Người bán được đánh giá bởi các yếu tố :
    #Số lần bán đã thực hiện
    #Thời gian trung bình cần thiết để người bán cung cấp cho người vận chuyển
    #Số tiền thực hiện trong tổng doanh thu
    #Đánh giá nhận được cho đơn hàng
##Kết hợp bảng order_item với bảng order theo order_id
df_seller_seg=pd.merge(df_items[['order_id','seller_id','price']],df_orders[['order_id','order_purchase_timestamp','order_delivered_carrier_date']],on='order_id')

##Kết hợp điểm đánh giá của đơn đặt hàng với order_id
df_seller_seg=pd.merge(df_seller_seg,df_reviews[['order_id','review_score']],on='order_id')

##converting dates to datetime format
df_seller_seg['order_purchase_timestamp']=pd.to_datetime(df_seller_seg['order_purchase_timestamp']).dt.date
df_seller_seg['order_delivered_carrier_date']=pd.to_datetime(df_seller_seg['order_delivered_carrier_date']).dt.date
df_seller_seg['days_to_del']=(df_seller_seg['order_delivered_carrier_date']-df_seller_seg['order_purchase_timestamp']).dt.days

## Loại bỏ cột dư thừa
df_seller_seg.drop(['order_purchase_timestamp','order_delivered_carrier_date'],axis=1,inplace=True)

## Điền các giá trị còn thiếu bằng giá trị trung bình
df_seller_seg['days_to_del'].fillna(df_seller_seg['days_to_del'].mean(),inplace=True)

## Loại bỏ những cột có giá trị bằng 0
df_seller_seg_1=df_seller_seg[df_seller_seg['days_to_del']>=0]

## Tạo bảng gồm các cột tổng số lượng bán (Tot_sales) ,tổng tiền (Tot_amount), trung bình review, thời gian trung bình để giao hàng cho người vận chuyển
df_seller_seg_2=df_seller_seg_1.groupby('seller_id').agg({'order_id':'count','price':'sum','review_score':'mean','days_to_del':'mean'})
df_seller_seg_2.columns=['Tot_sales','Tot_amount','Avg_review','Avg_delivery']

## Điền các giá trị còn thiếu bằng giá trị trung bình
df_seller_seg_2['Avg_delivery'].fillna(df_seller_seg_2['Avg_delivery'].mean(),inplace=True)

##splitting into quartiles
quantiles = df_seller_seg_2.quantile(q=[0.25,0.5,0.75])
quantiles.to_dict()
print('seller seg 2 : \n', df_seller_seg_2)
print('quantiles: \n',quantiles)

# thống kê phân tích các đặc trưng
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
df_seller_seg_2.to_csv(r'df_seller_2.csv', index = False)
### Có bao nhiêu người bán trong mỗi phân khúc ?
print("Best Sellers: ",len(df_seller_seg_2[df_seller_seg_2['SellerScore']=='4444']))
print('highest Reviewed Sellers: ',len(df_seller_seg_2[df_seller_seg_2['Avg_review_Quartile']==4]))
print("Fastest delivering sellers: ",len(df_seller_seg_2[df_seller_seg_2['Avg_delivery_Quartile']==4]))

## Số Người bán hàng có lượt review cao nhất cà giao nhanh nhất là?
Fastest_delivering =df_seller_seg_2[df_seller_seg_2['Avg_delivery_Quartile']==4].reset_index()
highest_Reviewed=df_seller_seg_2[df_seller_seg_2['Avg_review_Quartile']==4].reset_index()
person_fast_highest=len(set(list(Fastest_delivering['seller_id'])).intersection(highest_Reviewed['seller_id'].tolist()))
print("highest Reviewed & Fastest delivering",person_fast_highest)

############################## Dự đoán khách mua hàng ######################################

## Kết hợp các bảng orders , reviews và dữ liệu thanh toán với tập dữ liệu khách hàng và loại bỏ đi cột không mong muốn
df1=pd.merge(df_customer.drop(columns=['customer_zip_code_prefix']),df_orders[['customer_id','order_id','order_purchase_timestamp']],on='customer_id')
df2=pd.merge(df1,df_reviews[['order_id','review_score']],on='order_id')
paid=df_payments[['order_id','payment_value']].groupby('order_id').sum().reset_index()
df3=pd.merge(df2,paid,on='order_id')

#Lấy từ tập dữ liệu 180 ngày(6 tháng) kể từ ngày khách hàng mua gần nhất. Và dự đóan 1 khách hàng có mua hàng trong khoảng thời gian đó không
## Tạo ngày mua ở định dạng datetime
df3['order_purchase_timestamp']=pd.to_datetime(df3['order_purchase_timestamp']).dt.date
number_of_days_for_purchase=180
max_date_in_data= df3['order_purchase_timestamp'].max()
data_split_date=max_date_in_data-dt.timedelta(days=number_of_days_for_purchase)
df_full=df3[df3['order_purchase_timestamp']<=data_split_date]
df_last=df3[df3['order_purchase_timestamp']>data_split_date]
df_last_180=pd.dataframe({'customer_unique_id':df3['customer_unique_id'].values.tolist()})
df_last_180=df_last_180.merge(df_last.groupby(['customer_unique_id'])['payment_value'].sum().reset_index(),how='outer',on='customer_unique_id')
df_last_180.fillna(0,inplace=true)
df_last_180['purchased']=np.where(df_last_180['payment_value']>0, 1,0)
df_last_180.head()
print("tập dữ liệu trong khoảng 180 ngày gần nhất: \n",df_last_180)
df_last_180.to_csv(r'df_last_180.csv', index = false)


##################################### Tính đặc trưng ###############################
## Tổng tiền cho mỗi khách
tot_Amount=df_full.groupby('customer_unique_id')['payment_value'].sum().reset_index().rename(columns={'payment_value':'total_amount'})

## Đánh giá trung bình đưa ra
avg_review=df_full.groupby('customer_unique_id')['review_score'].mean().reset_index().rename(columns={'review_score':'avg_review'})

## Tháng tính từ lần đầu tiên mua tới hôm nay
min_max_date=df_full.groupby('customer_unique_id')['order_purchase_timestamp'].agg([min,max])
min_max_date['diff_first_today']=(dt.datetime.today().date()-min_max_date['min']).dt.days
## Tháng từ lần đầu tiên mua tới lần cuối cùng
min_max_date['max']=pd.to_datetime(min_max_date['max'])
min_max_date['min']=pd.to_datetime(min_max_date['min'])
min_max_date['diff_first_last']=(min_max_date['max']-min_max_date['min']).dt.days

## Doanh số gần đây
max_date=df_full['order_purchase_timestamp'].max()
min_max_date['recency']=(np.datetime64(max_date)-min_max_date['max'])/np.timedelta64(1, 'M')

## Tần suất bán hàng
frequency=df_full.groupby('customer_unique_id')['order_id'].count().reset_index().rename(columns={'order_id':'frequency'})

## Kết hợp tất cả các đặc trưng
dataset=pd.merge(tot_Amount,avg_review,on='customer_unique_id')
dataset=pd.merge(dataset,min_max_date,on='customer_unique_id')
dataset=pd.merge(dataset,frequency,on='customer_unique_id')
dataset=pd.merge(dataset,df_full[['customer_unique_id','customer_city','customer_state']],on='customer_unique_id')
dataset.drop(['min','max'],axis=1,inplace=True)

### Mã hóa cột customer_city và customer_state
encoder=LabelEncoder()
dataset['customer_city']=encoder.fit_transform(dataset['customer_city'])
dataset['customer_state']=encoder.fit_transform(dataset['customer_state'])
dataset.to_csv(r'dataset.csv', index = False)
##Kết hợp dataset với bộ df_last_180 tạo bên trên theo customer_unique_id sau đó loại bỏ đi cột customer_unique_id
dataset_full=dataset.merge(df_last_180[['customer_unique_id','purchased']],on='customer_unique_id')
dataset_full.drop(columns='customer_unique_id',inplace=True)
dataset_full.to_csv(r'dataset_full.csv', index = False)
##Phân tách và kiểm tra tập dữ liệu
X_train,X_test,y_train,y_test=train_test_split(dataset_full.iloc[:,:-1],dataset_full.iloc[:,-1], test_size=0.2, random_state=31)

print('Xtrain= \n',X_train)
X_train.to_csv(r'xtrain.csv', index = False)
print('XTest= \n',X_test)
X_test.to_csv(r'xtest.csv', index = False)
print('ytrain= \n',y_train)
y_train.to_csv(r'ytrain.csv', index = False)
print('Ytest=\n',y_test)
y_test.to_csv(r'ytest.csv', index = False)

