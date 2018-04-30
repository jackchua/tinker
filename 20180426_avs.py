import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# read trx data
colnames = 'id,chain,dept,category,company,brand,date,productsize,productmeasure,purchasequantity,purchaseamount'.split(',')
loc_dir = '/Users/jachua/data/acquirevaluedshoppers'
trx_loc = os.path.join(loc_dir, 'transactions_sampled.csv.gz')
trx = pd.read_csv(trx_loc, compression='gzip', names=colnames, sep=',')

# initialize encoders
label_encoders = {}
one_hot_encoders = {}

####################################################################
# data engineering: remove refunds and negative amount values
# TODO: remove outliers; will affect our regression
# TODO: plot of quantity versus demand for a given product over time
trx = trx.loc[trx['purchaseamount'] > 0.0, ]
trx = trx.loc[trx['purchasequantity'] > 0.0, ]
trx['unitpurchaseamount'] = trx['purchaseamount']/trx['purchasequantity']

# encode the product measure
lbl_enc = LabelEncoder()
trx['productmeasure'] = lbl_enc.fit_transform(trx['productmeasure'])
label_encoders['productmeasure'] = lbl_enc

# get the top 50 products by quantity
trx_counts = trx.groupby('category').agg({'purchasequantity':'sum'})
trx_counts = trx_counts.sort_values(by='purchasequantity', ascending=False)[0:50]
trx_counts['category'] = trx_counts.index
trx_counts['sum_purchasequantity'] = trx_counts['purchasequantity']
trx = pd.merge(trx, trx_counts[['category','sum_purchasequantity']], on='category', how='inner')

len(trx['date'].unique()) #513 days

# distribution of trx counts per product
prod_counts = trx[['category','chain']].groupby('category').count()[['chain']]
prod_counts.hist(bins=100)
plt.title('Distribution of trx counts per product')
plt.show()

# price variation in a product
price_var = trx[['category','purchaseamount', 'purchasequantity']].groupby('category').agg(
    {
        'purchaseamount':['mean','std'],
        'purchasequantity':['mean','std']
    }
)
price_var['purchaseamount','mean'].hist(bins=50)
plt.show()
price_var['purchaseamount','std'].plot(kind='kde')
plt.show()

# problem is - everything has a power law
# you have to price products without much data with products that have more data
# three layer hierarchical model:  category < dept
# oops! category is the product category. so two levels at most.

# convert to sparse dataframe
# lots of dummy variables, have to know how to deal with it
# apart from pd.get_dummies, there are other ways to do it (like with scipy or sklearn's
# feature vectorizers

# this will make scoring much easier
data = pd.SparseDataFrame(trx)
cat_vars = ['dept','category','brand','productmeasure']

# then accumulate all the codes and create one hot encoding
for cat_var in cat_vars:
    ohe = OneHotEncoder(categorical_features='all', sparse=True)
    dums = ohe.fit_transform(data[cat_vars])
    data = pd.concat([data, dums], axis=1)

for var in cat_vars+['date','sum_purchasequantity','purchaseamount','company','chain']:
    del data[var]

####################################################################
# part 2: learning elasticities
# does size have anything to do with elasticity?
# volume? company/branding?

# solution 1: nonparametric (random forest)
#
# produce a demand forecast
rf = RandomForestRegressor(n_jobs=4)
y = data['purchasequantity']
X = data.loc[:, data.columns != 'purchasequantity']
n_train = int(X.shape[0]*0.70)
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=n_train,random_state=4)
rf.fit(X_train, y_train)

y_hat = rf.predict(X_test)
errors = pd.DataFrame(list(y_hat - y_test.values), columns=['errors'])
errors = errors[errors['errors']<10]
errors = errors[errors['errors']>-10]

rmse = np.sqrt(np.mean(np.square(errors)))
mae = np.mean(np.abs(errors))
pd.DataFrame(errors).hist(bins=50)
plt.show()

# for each product, simulate the demand for a -10% to +10% increase
# EG: what should chain 100 price for each brand of product 6320?
price_steps = np.arange(-0.1, 0.1, 0.01)

# brand=14760, category=516, chain=100
dp = 0.05
trx.loc[trx['category']==516,]['brand'].unique()

