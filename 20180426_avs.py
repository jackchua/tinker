import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
import os
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def prepare_data(data):
    data = pd.SparseDataFrame(data)
    y = data['purchasequantity']
    X = data.loc[:, data.columns != 'purchasequantity']
    return X,y

def train_or_score(
        rows, label_vars=[], cat_vars=[], label_encoders={}, one_hot_encoder=None, score=False
):
    # these are variables taht need to be turned into categoricals
    if len(label_vars)>0:
        for l in label_vars:
            if score==False:
                le = LabelEncoder()
                rows[l] = le.fit_transform(rows[[l]])
                label_encoders[l] = le
            else:
                rows[l] = label_encoders[l].transform(rows[[l]])

    # produce one hot encoded features for each categorical
    # utilize sparse data structures
    if len(cat_vars+label_vars)>0:
        rows = pd.SparseDataFrame(rows)
        if score==False:
            one_hot_encoder = OneHotEncoder(categorical_features='all', sparse=True)
            dums = pd.SparseDataFrame(one_hot_encoder.fit_transform(rows[cat_vars+label_vars]))
        else:
            dums = pd.SparseDataFrame(one_hot_encoder.transform(rows[cat_vars+label_vars]))
        dums = dums.fillna(0.0)
        rows = pd.concat([rows, dums], axis=1)

    # drop the original categoricals
    for c in cat_vars+label_vars:
        rows = rows.loc[:, rows.columns != c]

    return (rows, label_encoders, one_hot_encoder)

####################################################################
# data engineering: remove refunds and negative amount values
# TODO: remove outliers; will affect our regression
# TODO: plot of quantity versus demand for a given product over time
colnames = 'id,chain,dept,category,company,brand,date,productsize,productmeasure,purchasequantity,purchaseamount'.split(',')
loc_dir = '/Users/jachua/data/acquirevaluedshoppers'
trx_loc = os.path.join(loc_dir, 'transactions_sampled.csv.gz')
trx = pd.read_csv(trx_loc, compression='gzip', names=colnames, sep=',')
trx = trx.loc[trx['purchaseamount'] > 0.0, ]
trx = trx.loc[trx['purchasequantity'] > 0.0, ]
trx['unitpurchaseamount'] = trx['purchaseamount']/trx['purchasequantity']
vars_to_delete = ['id','date','purchaseamount']
for v in vars_to_delete:
    trx = trx.loc[:, trx.columns != v]

# get the top 50 products by quantity
trx_counts = trx.groupby('category').agg({'purchasequantity':'sum'})
trx_counts = trx_counts.sort_values(by='purchasequantity', ascending=False)[0:50]
trx_counts['category'] = trx_counts.index
trx_counts['sum_purchasequantity'] = trx_counts['purchasequantity']
trx = pd.merge(trx, trx_counts[['category']], on='category', how='inner')

# len(trx['date'].unique()) #513 days

# distribution of trx counts per product
# prod_counts = trx[['category','chain']].groupby('category').count()[['chain']]
# prod_counts.hist(bins=100)
# plt.title('Distribution of trx counts per product')
# plt.show()
#
# # price variation in a product
# price_var = trx[['category','purchaseamount', 'purchasequantity']].groupby('category').agg(
#     {
#         'purchaseamount':['mean','std'],
#         'purchasequantity':['mean','std']
#     }
# )
# price_var['purchaseamount','mean'].hist(bins=50)
# plt.show()
# price_var['purchaseamount','std'].plot(kind='kde')
# plt.show()

# problem is - everything has a power law
# you have to price products without much data with products that have more data
# three layer hierarchical model:  category < dept
# oops! category is the product category. so two levels at most.

data = pd.SparseDataFrame(trx)
y = data['purchasequantity']
X = data.loc[:, data.columns != 'purchasequantity']
label_vars = ['productmeasure']
cat_vars = ['dept','category','brand']
X, label_encoders, one_hot_encoder = train_or_score(
    X, label_vars=label_vars, cat_vars=cat_vars
)

####################################################################
# part 2: learning elasticities
# does size have anything to do with elasticity?
# volume? company/branding?

# SOLUTION 1: nonparametric (random forest)
# produce a demand forecast
rf = RandomForestRegressor(n_jobs=4)
n_train = int(X.shape[0]*0.70)
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=n_train,random_state=4)
rf.fit(X_train, y_train)

y_hat = rf.predict(X_test)
errors = pd.DataFrame(list(y_hat - y_test.values), columns=['errors'])
errors = errors[errors['errors'] < 10]
errors = errors[errors['errors'] > -10]

rmse = np.sqrt(np.mean(np.square(errors)))
mae = np.mean(np.abs(errors))
pd.DataFrame(errors).hist(bins=50)
plt.show()

# for each product, simulate the demand for a -10% to +10% increase
# EG: what should chain 100 price for each brand of product 6320?
#     brand=14760, category=516, chain=100
# TODO: try to cross-validate the best model
mean_unitpurchaseamount = data.loc[data.category==6320,'unitpurchaseamount'].mean()
price_steps = np.arange(-0.1, 0.1, 0.005)
chain = [100]
dept = [63]
category = [6320]
company = [1068826767]
brand = [12475]
productsize = data[data.category==6320]['productsize'].unique()
productmeasure = data[data.category==6320]['productmeasure'].unique()
purchasequantity = [1]
unitpurchaseamount = [mean_unitpurchaseamount*(1-perc) for perc in price_steps]

lists = [chain, dept, category, company, brand, productsize, productmeasure, purchasequantity, unitpurchaseamount]
prices = pd.DataFrame(list(product(*lists)), columns=[
    'chain','dept','category','company','brand','productsize','productmeasure','purchasequantity','unitpurchaseamount'
])
prices_X, prices_y = prepare_data(prices)
prices_X, label_encoders, one_hot_encoder = train_or_score(
    prices_X, label_vars=label_vars, cat_vars=cat_vars,
    label_encoders=label_encoders, one_hot_encoder=one_hot_encoder,
    score=True
)
prices_y_hat = rf.predict(prices_X)
prices['pred'] = prices_y_hat
prices['group'] = prices['productsize'].astype('str') + '-' + prices['productmeasure']
prices['pred_ewm'] = prices.groupby('group').apply(lambda x: x[['pred']].ewm(span=10).mean())

# use the smoothed prediction to determine the best price
optimal_prices = prices.loc[prices.groupby(["productmeasure", "productsize",'group'])['pred'].idxmax()]
optimal_prices['perc'] = (optimal_prices['unitpurchaseamount'] - mean_unitpurchaseamount)/mean_unitpurchaseamount

# TODO: just run a regression
# TODO: plot some curves to show estimated elasticity
# should see for this product that bulk items should be (on average) increased in price, while smaller quantities should be decreased
fig, ax = plt.subplots()
for name, group in prices.groupby('group'):
    group.plot(x='unitpurchaseamount',y='pred_ewm', ax=ax, legend=True, label=name)
plt.show()