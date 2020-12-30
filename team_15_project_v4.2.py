# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 20:29:17 2020

@author: caesa
"""


from datetime import datetime, timedelta, date
from  matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score,cross_validate 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import validation_curve, KFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
# explore adaboost ensemble tree depth effect on performance
from numpy import mean
from numpy import std
# from sklearn.ensemble import GradientBoostingClassifier
# import xgboost as xgb
# from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import statistics as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# import re

#Read data into file
data_raw = pd.read_excel("C:/Users/caesa/Documents/UCI/Fall 2020/BANA 212/Project/global_superstore/Global_Superstore2.xlsx")

raw_data = data_raw.copy()

#Clean up column names
temp_col = []

for ele in raw_data.columns:
    col = ele.lower().split(' ')
    # print(col)
    for words in col:
        new_name = ""
        new_name += "_".join(col)
    temp_col.append(new_name)
        
raw_data.columns = temp_col
raw_data = raw_data.rename(columns = {"sub-category":"sub_category"})
    
del [col, ele, new_name, temp_col, words]

#change data type from string to date
raw_data['order_date'] = pd.to_datetime(raw_data['order_date']).dt.date
raw_data['year']= raw_data['order_date'].apply(lambda x: x.year)
    
#identify threshold between training and testing data
cutoff_date = date(2014,10,1)
min_date = raw_data['order_date'].min()
max_date = raw_data['order_date'].max()

# #calculate time since cutoff
# raw_data['recency'] = raw_data['order_date'] - cutoff_date


#create accurate customer id. Start by deleting inaccurate one
customer = pd.DataFrame(raw_data['customer_name'].unique())
customer['customer_id'] = [i for i in range(1,len(customer)+1)]
customer.columns = ['customer_name', 'customer_id']
customer_dict = dict(zip(customer.iloc[:,0],customer.iloc[:,1]))

raw_data['customer_id'] = raw_data.customer_name.apply(lambda x: customer_dict[x])

#Our data starts 2011 and ends 2014. Cutoff is when we are training / tresting
tx_prior = raw_data[(raw_data.order_date < cutoff_date) & (raw_data.order_date >= min_date)].reset_index(drop=True)
tx_next = raw_data[(raw_data.order_date >= cutoff_date) & (raw_data.order_date < max_date)].reset_index(drop=True)

del [min_date, max_date]



            ###OPTIONAL SECTION: OUTLIERS
#identifying top sales by order_id so that we can filter out outliers
outliers_total = tx_prior.groupby(['year','order_id'])['sales'].agg(
    ['sum','mean']).sort_values(by = 'sum', ascending = False).reset_index()

#Box plot of sales by order_id separated by year
fig, ax = plt.subplots(figsize=(16,9))
ax = sns.boxplot(x = 'year', y = 'sum', data = outliers_total)
plt.title('Number of orders per user by year')

#without taking year into account, remove the top 5% of sales, consider those outliers
outliers = tx_prior.groupby('order_id')['sales'].agg(
    ['sum','mean']).sort_values(by = 'sum', ascending = False).reset_index()

# outliers = tx_prior.groupby('order_id')['sales'].sum().sort_values(ascending = False)
outlier_limit = int(outliers.quantile(.95, axis = 0)[0])
outliers_order_id = outliers[outliers['sum'] > outlier_limit]



# Identify order count to filter out non-recurring users
outliers_tx = tx_prior.groupby(['customer_id','year'])['order_id'].count().reset_index()
# outliers_tx.reset_index()
outliers_tx.columns = ['customer_id', 'year', 'num_orders']
outliers_tx = outliers_tx.pivot(index = 'customer_id', columns = 'year', values = 'num_orders')
    
#illustrates the order distribution by year (or should we filter out based on all year's total order count?)
quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
# outliers_tx.quantile(quantiles)

#Customer Order count distribution by year    
    # year  2011   2012  2013   2014
    # 0.01   1.0   2.00   4.0   3.00
    # 0.05   3.0   4.00   6.0   5.00
    # 0.25   7.0   8.00  11.0   9.00
    # 0.50   9.0  12.00  15.0  13.00
    # 0.75  14.0  16.00  20.0  17.00
    # 0.90  17.0  20.00  24.0  22.00
    # 0.95  20.0  23.00  28.0  25.00
    # 0.99  25.0  28.07  34.0  30.07

#some orders remain above 3 std 
temp_years = outliers_tx.reset_index().melt(id_vars = ['customer_id'])
sns.boxplot(x = 'year', y = 'value', data = temp_years)
plt.title('Number of orders per user by year')

temp_overall = outliers_tx.reset_index().melt(id_vars = ['customer_id']).groupby('customer_id')['value'].sum()
temp_overall = temp_overall.reset_index().rename(columns = {'value':'num_orders'})
# temp_overall.num_orders.quantile(quantiles)

# Customer's total Order Count frequency distribution
    # 0.01    29.0
    # 0.05    34.7
    # 0.25    44.0
    # 0.50    52.0
    # 0.75    60.0
    # 0.90    67.0
    # 0.95    72.3
    # 0.99    80.0    
sns.displot(temp_overall, x = 'num_orders')
plt.title('Total customer orders over last 4 years')
        #Overall most customers have over 40 orders. seems to be bell shaped distribution
del [temp_years, temp_overall, quantiles,outliers_tx ]

# remove all orders (each transaction) associated with an outlier order's id number, "~" means NOT
tx_prior = tx_prior[~tx_prior.order_id.isin(outliers_order_id.iloc[:,0])]
del [outliers, outlier_limit, outliers_order_id]




                #Identifying Days since Cutoff when transaction Occurs
#identifying days between orders adjacent to cutoff date
#Group orders by order_id
temp_order_data = tx_prior.groupby(['order_id','year','segment'])['sales'].sum().reset_index()
sns.set_theme(style="darkgrid")
ax = sns.catplot(x = 'year', y = 'sales', hue = 'segment', data = temp_order_data, kind = 'violin')
plt.title('Sales Distribution by orders')

#identify each user's first and last orders based on cutoff date. Last oder before cutoff, first order after cutoff date
tx_last_order = tx_prior.groupby('customer_id')['order_date'].max().reset_index()
tx_first_order = tx_next.groupby('customer_id')['order_date'].min().reset_index()

#create unique list indicating days since last purchase (before cutoff date) and next purchase
cust_orders = tx_last_order.merge(tx_first_order, on = 'customer_id', how = 'left')

# len(tx_prior['customer_id'].unique())
# len(cust_orders['customer_id'].unique())

#Based on cutoff date, calculate the days between adjacent orders
cust_orders.columns = ['customer_id', 'prior_purchase', 'next_purchase']
cust_orders['days_next_order'] = cust_orders.next_purchase - cust_orders.prior_purchase
cust_orders['days_next_order'] = cust_orders['days_next_order'].dt.days
cust_orders['days_next_order'] = cust_orders['days_next_order'].fillna(999)
cust_orders['days_next_order'] = cust_orders['days_next_order'].astype(int)


                    ###RECENCY Calculation###

#Group customers based on their invoices to find their LAST order date and how recent was it
temp = tx_prior.groupby('customer_id')['order_date'].max().reset_index()

# Last applicable date is one day before data split (cutoff date)
last_app_date = cutoff_date - timedelta(days=1)
#calc how many days ago was the last order from cutoff date
temp['recency'] = (last_app_date - temp['order_date']).dt.days
temp.columns = ['customer_id', 'last_order_date', 'recency']

cust_orders = pd.merge(cust_orders, temp[['customer_id','recency']], on = 'customer_id')

#ELBOW PLOT to determine the number of clusters
# from sklearn.cluster import KMeans

#distortion avg squared distances from clusters to cluster center, i believe euclidean
#intertia: sum squared distances of sample to closest cluster center


def elbow_check(raw_data,cluster_attribute):
    
    temp_distortion = []
    cluster_range = range(1,10+1)
    
    for i in cluster_range:
        temp_kmeanmodel = KMeans(n_clusters = i, max_iter = 1000)
        temp_kmeanmodel.fit(raw_data[[cluster_attribute]])
        # temp_distortion.append(int(round(temp_kmeanmodel.inertia_,0)))
        temp_distortion.append(temp_kmeanmodel.inertia_)
        
    plt.figure(figsize=(10,8))
    plt.plot(cluster_range, temp_distortion, 'bx-')
    plt.xlabel('Clusters')
    plt.ylabel('Distortion')
    plt.title('{}: Elbow plot to determine optimal number of clusters'.format(cluster_attribute.upper()))
    plt.show()
    
cluster_attribute = 'recency'
cluster_group = 'recency_cluster'
elbow_check(cust_orders, cluster_attribute)

####Based on Elbow plot, classify each user by their cluster
kmeans = KMeans(n_clusters = 3)
kmeans.fit(cust_orders[[cluster_attribute]])
cust_orders[cluster_group] = kmeans.predict(cust_orders[[cluster_attribute]])

def order_cluster(data, cluster_name, attribute):
        
    data_new = data.groupby(cluster_name)[attribute].mean().reset_index()
    
    #sort clusters from high to low and reset value
    data_new = data_new.sort_values(by=attribute,ascending=False).reset_index(drop=True)

    data_new['index'] = data_new.index
    data = pd.merge(data,data_new[[cluster_name,'index']], on=cluster_name)
    # print(data.head(3))
    
    del data[cluster_name]
    data.rename(columns={"index":cluster_name},inplace = True)
    return data

cust_orders = order_cluster(cust_orders, cluster_group,cluster_attribute)







                    ###Frequency Calculation###
temp = tx_prior.groupby('customer_id')['order_date'].count().reset_index()
temp.columns = ['customer_id','frequency']

cust_orders = pd.merge(cust_orders,temp, on = 'customer_id', how = 'left')

# sns.displot(cust_orders, x = 'frequency')
# plt.title('Total customer orders over last 4 years')

#ELBOW PLOT to determine the number of clusters
cluster_attribute = 'frequency'
cluster_group = 'freq_cluster'
elbow_check(cust_orders, cluster_attribute)

####Based on Elbow plot, classify each user by their cluster
kmeans = KMeans(n_clusters = 3)
kmeans.fit(cust_orders[[cluster_attribute]])
cust_orders[cluster_group] = kmeans.predict(cust_orders[[cluster_attribute]])

cust_orders = order_cluster(cust_orders, cluster_group,cluster_attribute)






                    ###Monetary Value Calculation###
temp = tx_prior.groupby('customer_id')['sales'].sum().reset_index()

cust_orders = pd.merge(cust_orders,temp, on = 'customer_id', how = 'left')

# sns.displot(cust_orders, x = 'sales')
# plt.title('Customer sales over last 4 years')

#ELBOW PLOT to determine the number of clusters
cluster_attribute = 'sales'
cluster_group = 'sales_cluster'
elbow_check(cust_orders, cluster_attribute)

####Based on Elbow plot, classify each user by their cluster
kmeans = KMeans(n_clusters = 3)
kmeans.fit(cust_orders[[cluster_attribute]])
cust_orders[cluster_group] = kmeans.predict(cust_orders[[cluster_attribute]])

cust_orders = order_cluster(cust_orders, cluster_group,cluster_attribute)




                        #Overal RFM Value
cust_orders['rfm_score'] = cust_orders['sales_cluster'] + cust_orders['recency_cluster'] + cust_orders['freq_cluster']
# cust_orders.groupby('rfm_score')['recency','frequency','sales'].mean()

def rank_cust(x):
    if x <= 3:
        return 'Low_value'
    elif x <= 6:
        return 'Medium_value'
    else:
        return 'High_value'
cust_orders['cust_rank'] = cust_orders.rfm_score.apply(lambda x: rank_cust(x))

del [cluster_attribute, cluster_group, kmeans, last_app_date, temp, 
     temp_order_data]





                        #Days since invoice
###main driving input. Determins how many past transaction data we are evaluating
tracked_purchases = 10                        
                        
purch_hist = tx_prior.sort_values(['customer_id','order_date'], ascending = False)
purch_hist = purch_hist.groupby(['customer_id','order_date']).count().reset_index()

#Sort data so that we can filter on the x recent purchases
purch_hist = purch_hist.sort_values(by =['customer_id','order_date'], ascending = [True, True])
                        
#filter data and keep only relevant columns
for col in purch_hist.columns:
    if col not in ['customer_id', "order_date"]:
        del purch_hist[col]                        

#keep only latest customer orders history
purch_hist = purch_hist.groupby('customer_id').tail(tracked_purchases)

#identify each invoice from most to least recent (1 = most, x = less recent)
purch_hist['order_date'] = pd.to_datetime(purch_hist['order_date'])
purch_hist["periods"] = purch_hist.groupby("customer_id")["order_date"].rank(ascending=False)
purch_hist['periods'] = purch_hist['periods'].astype('int')
purch_hist['order_date'] = purch_hist['order_date'].dt.date

#shape the data so that every invoice period is listed as a column
purch_hist = purch_hist.pivot(index = 'customer_id', columns = 'periods', values = 'order_date').reset_index()

#find the difference, in days, between all subsequent invoices relative to the latest invoice
rel_purch_col = []
for i in range(1,tracked_purchases):
    name = 'day_diff_' + str(i)
    purch_hist[name] = (purch_hist[1]- purch_hist[i+1]).dt.days
    rel_purch_col.append(name)
        
#Diff Between subsequent dates. Used to find standard deviation and mean between EACh purchase
temp_mean_adj_dif = []
for i in range(1,tracked_purchases):
    name = 'adj_diff' + str(i)
    purch_hist[name] = (purch_hist[i]- purch_hist[i+1]).dt.days
    temp_mean_adj_dif.append(name)
    
#Mean and STD between each purchase
extra_col = ['std','mean']

temp_mean_and_std = purch_hist.melt(id_vars = 'customer_id', value_vars = temp_mean_adj_dif, value_name = 'order_pattern')
temp_mean_and_std.sort_values(by = ['customer_id'], inplace = True)
temp_mean_and_std = temp_mean_and_std.groupby('customer_id')['order_pattern'].agg(extra_col)



#merge cust order data with std and mean
purch_hist = pd.merge(purch_hist, temp_mean_and_std, on = 'customer_id', how = 'left')

del [col, i, name,temp_mean_adj_dif, temp_mean_and_std, tracked_purchases]









                    ###Merge Customer Data with Purchase History
                                     
                    
                                        
#Should we drop customer's who did not make another purchase / with insufficient data?



#creating this dynamic list of relevant columns for future merging
rel_purch_col += [x for x in extra_col]
        # rel_purch_col --> [day_diff_1', 'day_diff_2', 'day_diff_3', 'std', 'mean']
rel_purch_col.insert(0,'customer_id') 

data = pd.merge(cust_orders, purch_hist[rel_purch_col], on = 'customer_id', how = 'left')
data.columns
#identify customers whose next purchase days are within select ranges
data.days_next_order.describe()
        # count    795.000000
        # mean      89.079245
        # std      178.281398
        # min        1.000000
        # 25%       27.000000
        # 50%       50.000000
        # 75%       79.000000
        # max      999.000000
        # Name: days_next_order, dtype: float64

#GUESSED the categorization by period. See below. We should discuss
def class_prediction(x):
    if x < 30:
       return 2
    elif x < 60:
       return 1
    else:
       return 0

#Classify individuals based on next order period
data['group_next_purchase'] = data['days_next_order'].apply(class_prediction)

#removed so that the dummy variables will not be created for each date
del data['prior_purchase'], data['next_purchase']

#categorizes every categorical attribute into binary outcomes
data = pd.get_dummies(data)
##############################################################################
data['std'] = data['std'].astype('int')
data['mean'] = data['mean'].astype('int')
data['sales'] = data['sales'].astype('int')
##############################################################################
#display distribution of our assigned purchase interval classification
data.group_next_purchase.value_counts()/len(customer)
    # 0    0.550943
    # 1    0.296855
    # 2    0.152201
    # Name: group_next_purchase, dtype: float64

#heatmap to display attribute correlation to our assigned purchase interval classification
#dark red higher correlation, dark blue lower correlation
plt.figure(figsize = (30,20))
sns.heatmap(data.corr(), annot = True, vmin = -1, vmax = 1, 
            center = 0, cmap = 'coolwarm', linewidth = 3, linecolor = 'white')
plt.title('Data: Attribute Correlation')

del [ax, customer, cutoff_date, extra_col, rel_purch_col, 
     tx_first_order,tx_last_order, tx_next, tx_prior]

### Objects explained:
    # cust_orders: RFM Score by customer_id
    # customer_dict: dictionary with each customer assigned to a new ID
    # data: cust_orders + purch_his + interval_purchase_score (attribute:group_next_purchase)
    # raw_data: data imported from excel file + year column
    # purch_hist: Customer's recent order / purchase data


                    # ML Algorithms
# Algorithms / Ranked Goals
    # Random Forest
    # Naive Bayes
    # Logistic Regression
    # Optional:
    # Adaboost?
    # Ensemble?

#Note: remember to adjust clusters, transaction periods captures, days threshold for classification, seed
    #Clusters: 4, transaction periods captured: 10,  days threshold = [30,60,61]
# from sklearn.metrics import roc_curve, auc










# Identify models with high accuracy and consistency
irrelevant_train_cols = ['group_next_purchase','days_next_order','customer_id']
x, y = data[data.columns.difference(irrelevant_train_cols)], data.iloc[:,data.columns == 'group_next_purchase']
y = y.values[:,0]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 123)

kfold = KFold(n_splits=10, random_state=123)

cv_result_rdf = cross_val_score(RandomForestClassifier(),x_train,y_train, cv = kfold,scoring = "accuracy")
cv_result_ada = cross_val_score(AdaBoostClassifier(),x_train,y_train, cv = kfold,scoring = "accuracy")
cv_result_dt = cross_val_score(DecisionTreeClassifier(),x_train,y_train, cv = kfold,scoring = "accuracy")
cv_result_knn = cross_val_score(KNeighborsClassifier(),x_train,y_train, cv = kfold,scoring = "accuracy")
cv_result_nb = cross_val_score(GaussianNB(),x_train,y_train, cv = kfold,scoring = "accuracy")
cv_result_lr = cross_val_score(LogisticRegression(),x_train,y_train, cv = kfold,scoring = "accuracy")

cv_result_rdf
cv_result_ada
cv_result_dt
cv_result_knn
cv_result_nb
cv_result_lr

rdf_mean = cv_result_rdf.mean()
    # 0.5518
gnb_mean = cv_result_nb.mean()
    # 0.5847
dt_mean = cv_result_dt.mean()
    # 0.4930
knn_mean = cv_result_knn.mean()
    # 0.4120
ada_mean = cv_result_ada.mean()
    # 0.5377
lr_mean = cv_result_lr.mean()
    # 0.5690
    
cv_result_rdf.std()
    # 0.0751
cv_result_nb.std()
    # 0.07824
cv_result_dt.std()
    # 0.0705
cv_result_knn.std()
    # 0.0503
cv_result_ada.std()
    # 0.0541
cv_result_lr.std()
    # 0.0795
consider = ['RandomForest', 'GaussianNB', 'DecisionTree','KNN', 'AdaBoost', 'LogisticRegression']
overall_accuracy = [rdf_mean,gnb_mean,dt_mean,knn_mean,ada_mean,lr_mean ]

g = sns.barplot(x=consider, y=overall_accuracy)

plt.xticks(rotation=45)
plt.title('Avg Model Accuracy: 10-fold CV')


#rather than 0,2,3, we will identify them based on 'risk' profile
#2 is highest risk (no purchase within 2 months)
classes=['High', 'Medium', 'Low']

#check there aren't any NAN
data.apply(lambda x: x.isnull().sum())

# check class distribution
for ele in data.group_next_purchase.unique():
    count = len(data[data['group_next_purchase'] == ele])
    dist = round(count / len(data['group_next_purchase']),2)
    print('class {}: {} accounts for {}% of the {} column'.format(ele, count,dist,'group_next_purchase'))
    
# class 2: 233 accounts for 0.29% of the group_next_purchase column
# class 1: 256 accounts for 0.32% of the group_next_purchase column
# class 0: 306 accounts for 0.38% of the group_next_purchase column




                    ###Random Forest First Pass - Model with default parameters

#Randomforest to predict Y values
rdf_v1 = RandomForestClassifier()
rdf_v1.fit(x_train, y_train)


y_pred_rdf_v1 = rdf_v1.predict(x_test)

print('Random Forest: Baseline')
print(classification_report(y_test, y_pred_rdf_v1))


conf_matrix_rdf_v1= confusion_matrix(y_test, y_pred_rdf_v1)


# classes=['High', 'Medium', 'Low']
conf_matrix_rdf_v1 = confusion_matrix(y_test, y_pred_rdf_v1)
df_cm_rdf_v1 = pd.DataFrame(conf_matrix_rdf_v1, index=classes, columns=classes)

ax = sns.heatmap(df_cm_rdf_v1, cmap='Blues', annot=True)
plt.title('RDF Baseline: Risk Profile')


#class Distribution
# class 2: 233 accounts for 0.29% of the group_next_purchase column
# class 1: 256 accounts for 0.32% of the group_next_purchase column
# class 0: 306 accounts for 0.38% of the group_next_purchase column

# Random Forest: Baseline
#               precision    recall  f1-score   support

#            0       0.79      0.54      0.64        68
#            1       0.49      0.49      0.49        51
#            2       0.52      0.80      0.63        40

#     accuracy                           0.59       159
#    macro avg       0.60      0.61      0.59       159
# weighted avg       0.63      0.59      0.59       159





                    ### RANDOM FOREST: Second Pass - Feature Selection
                    
#feature selection
top_features = pd.DataFrame({'Features':x_train.columns,'Importance':rdf_v1.feature_importances_})
top_features.sort_values('Importance', ascending = False, inplace = True)
top_features.reset_index(drop = True, inplace = True)

top_features['importance_cum_sum'] = top_features.Importance.cumsum(axis = 0)

#select from 'top_features' all which yield up to this %
feature_threshold = 0.90
top_features['keep'] = top_features.importance_cum_sum.apply(lambda row: True if row < feature_threshold else False)
worst_features = top_features.loc[top_features['keep'] == False]['Features'].to_list()
                  
#identify training and testing data

irrelevant_train_cols += worst_features
x, y = data[data.columns.difference(irrelevant_train_cols)], data.iloc[:,data.columns == 'group_next_purchase']
y = y.values[:,0]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 123)

best_features = x.columns

sns.barplot(data = top_features, y = 'Features', x = 'Importance')

fig, ax = plt.subplots(figsize=(16,9))

#Randomforest to predict Y values
rdf_v2 = RandomForestClassifier()
rdf_v2.fit(x_train, y_train)


y_pred_rdf_v2 = rdf_v2.predict(x_test)

print('Random Forest: Feature Selection')
print(classification_report(y_test, y_pred_rdf_v2))


conf_matrix_rdf_v2= confusion_matrix(y_test, y_pred_rdf_v2)


df_cm_rdf_v2 = pd.DataFrame(conf_matrix_rdf_v2, index=classes, columns=classes)

ax = sns.heatmap(df_cm_rdf_v2, cmap='Blues', annot=True)
plt.title('RDF_v2 Feature: Risk Profile')


# Random Forest: Feature Selection
#               precision    recall  f1-score   support

#            0       0.78      0.53      0.63        68
#            1       0.48      0.51      0.50        51
#            2       0.51      0.75      0.61        40

#     accuracy                           0.58       159
#    macro avg       0.59      0.60      0.58       159
# weighted avg       0.62      0.58      0.58       159

#removing features actually increased overall accuracy















             ### RANDOM FOREST: Third Pass - gridsearch for optimal parameters + Feature selection
##search for best number of trees
param_range = np.arange(0,2000,100)
train_score, test_score = validation_curve(
                                RandomForestClassifier(),
                                X = x_train, y = y_train, 
                                param_name = 'n_estimators',
                                scoring = 'accuracy',
                                param_range =  param_range,
                                cv = 3, n_jobs = -1)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_score, axis=1)
train_std = np.std(train_score, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_score, axis=1)
test_std = np.std(test_score, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
##plot shows n-estimator has greatest gains 0-30 trees. Maximum appears to be ~1800
    


# hyper parameter tuning#
parameters = {'n_estimators' : [500]
              , 'criterion' : ('gini', 'entropy')
              , 'max_features' : ('auto', 'sqrt')
              , 'min_samples_split': (2,4,6,10)
              , 'min_samples_leaf': [1, 2, 4]
              , 'bootstrap' : (True, False)
              #,'min_weight_fraction_leaf' : (0.0,0.1,0.2,0.3)
              }

grid_search = GridSearchCV(RandomForestClassifier(random_state = 123)
                        , param_grid = parameters
                        , verbose = 1, n_jobs =-1
                        , cv = 10)

grid_search = grid_search.fit(x_train, y_train)
accuracy = grid_search.best_score_
best_params = grid_search.best_params_
best_params_rdf = best_params
best_params
accuracy

# Feature + Parameter

rdf_v3 = RandomForestClassifier(
    n_estimators = 500,
    min_samples_split = 10, min_samples_leaf = 4,
    max_features = 'auto', criterion = 'entropy',
    bootstrap = True, random_state = 123)
rdf_v3.fit(x_train, y_train)

y_pred_rdf_v3 = rdf_v3.predict(x_test)

print('Random Forest: Feature + Parameter')
print(classification_report(y_test, y_pred_rdf_v3))
conf_matrix_rdf_v3= confusion_matrix(y_test, y_pred_rdf_v3)

df_cm_rdf_v3 = pd.DataFrame(conf_matrix_rdf_v3, index=classes, columns=classes)

ax = sns.heatmap(df_cm_rdf_v3, cmap='YlOrBr', annot=True)
plt.title('RDF Forest: Feature + Parameter')

# Random Forest: Feature + Parameter
#               precision    recall  f1-score   support

#            0       0.84      0.71      0.77        72
#            1       0.45      0.30      0.36        50
#            2       0.43      0.76      0.55        37

#     accuracy                           0.59       159
#    macro avg       0.57      0.59      0.56       159
# weighted avg       0.62      0.59      0.59       159


#Features decreased accuracy
#Parameters increased overall accuracy, recall, and precision



              ###Naive BayesFirst Pass - Model with default parameters
#identify training and testing data
# remove_cols = ['group_next_purchase','days_next_order']



irrelevant_train_cols = ['group_next_purchase','days_next_order','customer_id']
x, y = data[data.columns.difference(irrelevant_train_cols)], data.iloc[:,data.columns == 'group_next_purchase']
y = y.values[:,0]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 123)


gnb = GaussianNB(priors = None)
gnb.fit(x_train,y_train)
y_pred_gnb_v1 = gnb.predict(x_test)



#graph the predictions for the testing dataset
high = 0
medium = 0
low = 0
categories = ["<1 Month", "1-2 Months","Over 2 Months"]
predictions = []

#categorize each prediction for bar graph
for group in y_pred_gnb_v1:
    if group == 0:
        low +=1
    elif group == 1:
        medium +=1
    else:
        high +=1

#append each category into a list and print the number of entries in each category
predictions.append(high)
predictions.append(medium)
predictions.append(low)
        
print("The number of low risk (<1 Month) predictions is " + str(high))
print("The number of Medium risk (1-2 Month) predictions is " + str(medium))
print("The nubmer of high risk (>2 Month) predictions is " + str(low))

#create the bar graph
plt.bar(categories, predictions)
plt.xlabel("Customer Purchase Timing")
plt.ylabel("Number of Customers")
plt.title("Testing Model Prediction")


print('Gaussian Naive Bayes')
print(classification_report(y_test, y_pred_gnb_v1))
conf_matrix_gnb_v1= confusion_matrix(y_test, y_pred_gnb_v1)

df_cm_gnb_v1 = pd.DataFrame(conf_matrix_gnb_v1, index=classes, columns=classes)
fig, ax = plt.subplots(figsize=(10,8))
ax = sns.heatmap(df_cm_gnb_v1, cmap='rocket_r', annot=True)
plt.title('GNB: Risk Profile')

# Gaussian Naive Bayes
#               precision    recall  f1-score   support

#            0       0.83      0.74      0.78        72
#            1       0.45      0.10      0.16        50
#            2       0.44      1.00      0.61        37

#     accuracy                           0.60       159
#    macro avg       0.57      0.61      0.52       159
# weighted avg       0.62      0.60      0.55       159

#check class 0 (highest risk)
# np.count_nonzero(y_pred_gnb_v1 == 0)



              ### AdaBoost
sns.axes_style("white")
               
irrelevant_train_cols = ['group_next_purchase','days_next_order','customer_id']
x, y = data[data.columns.difference(irrelevant_train_cols)], data.iloc[:,data.columns == 'group_next_purchase']
y = y.values[:,0]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 123)       

#Adaboost baseline - Decision Tree
ada_1 = AdaBoostClassifier(
    base_estimator = DecisionTreeClassifier(max_depth = 1))
ada_1.fit(x_train, y_train)

y_pred_ada_v1 = ada_1.predict(x_test)

print('AdaBoost: Max Depth 1')
#Default: n_estimators = 50, learning_rate = 1
print(classification_report(y_test, y_pred_ada_v1))


conf_matrix_ada_v1 = confusion_matrix(y_test, y_pred_ada_v1)

df_cm_ada_v1 = pd.DataFrame(conf_matrix_ada_v1, index=classes, columns=classes)

ax = sns.heatmap(df_cm_ada_v1, cmap='rocket_r', annot=True)
plt.title('AdaBoost: Max Depth 1')
# AdaBoost: Max Depth 1
#               precision    recall  f1-score   support

#            0       0.75      0.62      0.68        72
#            1       0.34      0.32      0.33        50
#            2       0.42      0.59      0.49        37

#     accuracy                           0.52       159
#    macro avg       0.50      0.51      0.50       159
# weighted avg       0.55      0.52      0.53       159







#Adaboost: Max Depth = 2
ada_2 = AdaBoostClassifier(
    base_estimator = DecisionTreeClassifier(max_depth = 2,random_state = 123))
ada_2.fit(x_train, y_train)

y_pred_ada_v2 = ada_2.predict(x_test)

print('AdaBoost: Max Depth 2')
#Default: n_estimators = 50, learning_rate = 1
print(classification_report(y_test, y_pred_ada_v2))

conf_matrix_ada_v2 = confusion_matrix(y_test, y_pred_ada_v2)
df_cm_ada_v2 = pd.DataFrame(conf_matrix_ada_v2, index=classes, columns=classes)

ax = sns.heatmap(df_cm_ada_v2, cmap='rocket_r', annot=True)
plt.title('AdaBoost Max Depth 2')
# AdaBoost: Max Depth 2
#               precision    recall  f1-score   support

#            0       0.71      0.28      0.40        72
#            1       0.33      0.52      0.41        50
#            2       0.47      0.68      0.56        37

#     accuracy                           0.45       159
#    macro avg       0.51      0.49      0.45       159
# weighted avg       0.54      0.45      0.44       159





#Adaboost Parameters optimized

param_range = np.arange(100,500,25)
train_score, test_score = validation_curve(
                                AdaBoostClassifier(
                                    DecisionTreeClassifier(max_depth = 1, random_state = 123)
                                    ),
                                X = x_train, y = y_train, 
                                param_name = 'n_estimators',
                                scoring = 'accuracy',
                                param_range =  param_range,
                                cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_score, axis=1)
train_std = np.std(train_score, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_score, axis=1)
test_std = np.std(test_score, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With AdaBoost")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
##

       
              
# cv=KFold(n_splits=10,shuffle=True,random_state=1)

#adaboost parameter optimization

#estimate number of trees
max_depth_range = list(range(1, 20,1))
# List to store the average RMSE for each value of max_depth:
accuracy = []
for i in max_depth_range:

    boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth = i, random_state = 123))
    boost.fit(x_train, y_train)
    score = boost.score(x_test, y_test)
    accuracy.append(score)

print(accuracy)
ada_plot_v3 = sns.relplot(x = max_depth_range, y = accuracy, kind = 'line')
ada_plot_v3.set_xlabels('Max Depth')
ada_plot_v3.set_ylabels('Overall Accuracy')
plt.title('Adaboost: Decision Tree Optimal Max Depth ')


params_ada={'n_estimators':np.arange(100,350,50)
        ,'base_estimator__max_depth': [8]
        ,'learning_rate':[.001, 0.01, 0.1]
        ,"base_estimator__criterion" : ["gini", "entropy"]
        ,"base_estimator__splitter" : ["best", "random"]
        }


grid_search_ada=GridSearchCV(boost, params_ada, scoring='accuracy', n_jobs=-1)

grid_search_ada_v1 = grid_search_ada.fit(x_train, y_train)
accuracy_ada_v1 = grid_search_ada_v1.best_score_
best_params_ada = grid_search_ada_v1.best_params_
# {'learning_rate': 0.001, 'n_estimators': 100}

# List of values to try for max_depth:


ada_v3 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth = 8,random_state = 123, criterion = 'entropy', splitter = 'random'),
    n_estimators = 300,
    learning_rate = 0.001,
    )
ada_v3.fit(x_train, y_train)

y_pred_ada_v3 = ada_v3.predict(x_test)

print('AdaBoost: Parameter Tuning')
print(classification_report(y_test, y_pred_ada_v3))

# classes=['High', 'Medium', 'Low']
conf_matrix_ada_v3 = confusion_matrix(y_test, y_pred_ada_v3)
df_cm_ada_v3 = pd.DataFrame(conf_matrix_ada_v3, index=classes, columns=classes)

ax2 = sns.heatmap(df_cm_ada_v3, cmap='YlOrBr', annot=True)
plt.title('AdaBoost Parameter Tunning: Risk Profile')

# AdaBoost: Parameter Tuning
#               precision    recall  f1-score   support

#            0       0.84      0.65      0.73        72
#            1       0.39      0.36      0.37        50
#            2       0.46      0.70      0.55        37

#     accuracy                           0.57       159
#    macro avg       0.56      0.57      0.55       159
# weighted avg       0.61      0.57      0.58       159