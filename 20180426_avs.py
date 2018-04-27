import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# read trx data
colnames = 'id,chain,dept,category,company,brand,date,productsize,productmeasure,purchasequantity,purchaseamount'.split(',')
loc_dir = '/home/jack/data/acquirevaluedshoppers'
trx_loc = os.path.join(loc_dir, 'transactions_sampled.csv.gz')
trx = pd.read_csv(trx_loc, compression='gzip', names=colnames, sep=',')

####################################################################
# data engineering: remove refunds and negative amount values
trx = trx.loc[trx['purchaseamount'] > 0.0, ]
trx = trx.loc[trx['purchasequantity'] > 0.0, ]
trx_counts = trx.groupby('id').agg({'purchaseamount':'count'})
trx_counts = trx_counts.sort_values(by='purchaseamount', ascending=False)[0:50]
trx_counts['id'] = trx_counts.index
trx = pd.merge(trx, trx_counts[['id']], on='id', how='inner')

len(trx['date'].unique()) #513 days

# distribution of trx counts per product
prod_counts = trx[['id','chain']].groupby('id').count()[['chain']]
prod_counts = prod_counts[prod_counts['chain']<=10]
prod_counts.hist(bins=100)
plt.title('Distribution of trx counts per product')
plt.show()

# price variation in a product
price_var = trx[['id','purchaseamount', 'purchasequantity']].groupby('id').agg(
    {
        'purchaseamount':['mean','std'],
        'purchasequantity':['mean','std']
    }
)
price_var['purchaseamount','mean'].hist(bins=50)
price_var['purchaseamount','std'].hist(bins=50)

# problem is - everything has a power law
# you have to price products without much data with products that have more data
# three layer hierarchical model:  category < dept
# oops! category is the product category. so two levels at most.

# convert to sparse dataframe
# lots of dummy variables, have to know how to deal with it
trx = pd.SparseDataFrame(trx)
for cat_var in ['id','chain','dept','category','company','brand','productmeasure']:
    dums = pd.get_dummies(trx[cat_var],prefix=cat_var,sparse=True)
    trx = pd.concat([trx, dums], axis=1)
    del trx[cat_var+['date']]

####################################################################
# part 2: learning elasticities
# does size have anything to do with elasticity?
# volume? company/branding?

# solution 1: nonparametric (random forest)
#
# produce a demand forecast
rf = RandomForestRegressor()
y = trx['purchasequantity']
X = trx.loc[:, trx.columns != 'purchasequantity']
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

# for each product, given a rnage of prices around the product mean price (-10% to +10%), determine the best
# price

# first get all permutations of products
price_steps = np.arange(-0.1, 0.1, 0.01)
X[].drop_duplicates