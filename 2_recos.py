import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
# from pyfm import pylibfm
from fastFM import als
import pandas as pd
import os
from tffm import TFFMRegressor
import tensorflow as tf

_BAYES_ITER = 30
_ALGO = 'svd'
_FM_USE_CONTEXTUAL_FEATURES = True
_FM_N_ITER = 200
_FM_STEP_SIZE = 10

####################################################################################
# FM
####################################################################################

def loadData(filename,path="ml-100k/"):
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({ "user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)

def joinFeatures(data, userFilename="u.user", itemFilename="u.item"):
    data = pd.DataFrame(data)

    # join user data
    data['user_id'] = data['user_id'].astype(int)
    user_data = pd.read_csv(userFilename, sep="|", header=None, names=['user_id', 'age','gender','occupation','zipcode'])
    # user_data = user_data[['user_id','age']]
    data = pd.merge(data, user_data, how='left', on='user_id')

    # join item data
    data['movie_id'] = data['movie_id'].astype(int)
    item_data = pd.read_csv(itemFilename, sep="|", header=None, names=['movie_id','movie title','release_date','video_release_date','imdb_url','unknown','Action','Adventure','Animation','Children\'s','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'], encoding = "ISO-8859-1")
    data = pd.merge(data, item_data, how='left', on='movie_id')
    # replcae all nan's with zeros
    data = data.fillna(0.0)
    return data.to_dict('records')


# file_path = os.path.dirname(os.path.abspath(__file__))
file_path = '/home/jack/source/tinker/'
data_path = os.path.join(file_path, 'data/small_movielens/')

(train_data, y_train, train_users, train_items) = loadData("ua.base", path=data_path)
(test_data, y_test, test_users, test_items) = loadData("ua.test", path=data_path)

if _FM_USE_CONTEXTUAL_FEATURES:
    train_data = joinFeatures(train_data, data_path+'u.user', data_path+'u.item')
    test_data = joinFeatures(test_data, data_path+'u.user', data_path+'u.item')

v = DictVectorizer()
X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)

# model = TFFMRegressor(
#     order=2,
#     rank=2,
#     optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
#     n_epochs=1000,
#     batch_size=-1,
#     init_std=0.0001,
#     input_type='sparse'
# )
# model.fit(X_train, y_train, show_progress=True)

# print('MSE:{}'.format(mean_squared_error(y_test, model.predict(X_test))))
if _ALGO == 'fm':
    def fmcv(rank, l2_reg_w, l2_reg_V):
        fm = als.FMRegression(n_iter=10, init_stdev=0.0001, rank=int(rank), l2_reg_w=l2_reg_w, l2_reg_V=l2_reg_V)
        fm.fit(X_train, y_train)
        val = -mean_squared_error(fm.predict(X_test), y_test)
        return val

    fmBO = BayesianOptimization(
        fmcv,
        {
            'rank': (1,2),
            'l2_reg_w': (0.5, 1.5),
            'l2_reg_V': (0.5, 1.5)
        }
    )
    gp_params = {"alpha": 1e-5}
    fmBO.maximize(n_iter=_BAYES_ITER, **gp_params) #best: 1,1.5,1)
    print('FM: %f' % fmBO.res['max']['max_val'])

    #
    # fm.fit(X_train, y_train)
    #
    # rmse_train = []
    # rmse_test = []
    # r2_score_train = []
    # r2_score_test = []
    #
    # for i in range(1, _FM_N_ITER):
    #     print('Iteration: {}'.format(i))
    #     fm.fit(X_train, y_train, n_more_iter=_FM_STEP_SIZE)
    #     y_pred = fm.predict(X_test)
    #
    #     new_rmse_train = np.sqrt(mean_squared_error(fm.predict(X_train), y_train))
    #     new_r2_train = r2_score(fm.predict(X_train), y_train)
    #     new_rmse_test = np.sqrt(mean_squared_error(fm.predict(X_test), y_test))
    #     new_r2_test = r2_score(fm.predict(X_test), y_test)
    #
    #     rmse_train.append(new_rmse_train)
    #     rmse_test.append(new_rmse_test)
    #     r2_score_train.append(new_r2_train)
    #     r2_score_test.append(new_r2_test)
    #
    #     print('[Train] RMSE={},R2={}'.format(new_rmse_train, new_r2_train))
    #     print('[Test]  RMSE={},R2={}'.format(new_rmse_test, new_r2_test))


    # from matplotlib import pyplot as plt
    # fig, axes = plt.subplots(ncols=2, figsize=(15, 4))
    #
    # x = np.arange(1, _FM_N_ITER)*_FM_STEP_SIZE
    # with plt.style.context('fivethirtyeight'):
    #     axes[0].plot(x, rmse_train, label='RMSE-train', color='r', ls="--")
    #     axes[0].plot(x, rmse_test, label='RMSE-test', color='r')
    #     axes[1].plot(x, r2_score_train, label='R^2-train', color='b', ls="--")
    #     axes[1].plot(x, r2_score_test, label='R^2-test', color='b')
    # axes[0].set_ylabel('RMSE', color='r')
    # axes[1].set_ylabel('R^2', color='b')
    # axes[0].legend()
    # axes[1].legend()
    #
    # # fm = pylibfm.FM(num_factors=100, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
    # # fm.fit(X_train[:,0:100],y_train)
    #
    # # Evaluate
    # preds = fm.predict(X_test)
    # from sklearn.metrics import mean_squared_error
    # print("FM MSE: %.4f" % mean_squared_error(y_test,preds))

##################################################################################
# NMF
##################################################################################

if _ALGO == 'svd':
    from surprise import SVD
    from surprise import Dataset
    from surprise.model_selection import cross_validate

    # Load the movielens-100k dataset (download it if needed),
    data = Dataset.load_builtin('ml-100k')

    def svdcv(n_factors, lr_all, reg_all):
        svd = SVD(n_factors=int(n_factors), lr_all=lr_all, reg_all=reg_all)
        cv_score = cross_validate(svd, data, measures=['RMSE'], cv=2)
        return -(cv_score['test_rmse'].mean())

    svdBO = BayesianOptimization(
        svdcv,
        {
            'n_factors': (15,25),
            'lr_all': (0.01, 0.05),
            'reg_all': (0.05, 0.15)
        }
    )
    gp_params = {"alpha": 1e-5}

    # best: 0.0026, 2.42, 0.03; 0.0309, 20.43, 0.1
    svdBO.maximize(n_iter=_BAYES_ITER, **gp_params)
    print('SVD: %f' % svdBO.res['max']['max_val'])

##################################################################################
# NMF
##################################################################################
# assess: print out 3 highest movies against 3 highest recommendations for some people
fm = als.FMRegression(n_iter=100, init_stdev=0.0001, rank=1, l2_reg_w=1.5, l2_reg_V=1)
fm.fit(X_train, y_train)

# error analysis
import matplotlib.pyplot as plt
plt.style.use('ggplot')
y_est_train = fm.predict(X_train)
y_est = fm.predict(X_test)
ests = pd.DataFrame(list(zip(y_est, y_test)), columns=['pred','actual'])
ests['error'] = ests['pred'] - ests['actual']
ests['error'].hist(bins=20)
error_std = ests['error'].std()

plt.title('FM residuals')
plt.axvline(x=-2*error_std, color='black', linestyle='--')
plt.axvline(x=2*error_std, color='black', linestyle='--')
plt.show()

# sanity check for FM with features
# for user 1
# user_train_x = list(filter(lambda x: x[1]['user_id']=='1', enumerate(train_data)))
# user_train_idx = [x[0] for x in user_train_x]
# user_train_x = [x[1] for x in user_train_x]
# user_test_x = list(filter(lambda x: x[1]['user_id']=='1', enumerate(test_data)))
# user_test_idx = [x[0] for x in user_test_x]
# user_test_x = [x[1] for x in user_test_x]
#
# user_train_y = y_train[user_train_idx]
# user_test_y  = y_test[user_test_idx]
#
# v.transform(user_train_x)
dict_merge = lambda a,b: a.update(b) or a
sanity_check_train = v.inverse_transform(X_train)
sanity_check_train = [dict_merge(x, {'y': y_train[idx]}) for idx,x in enumerate(sanity_check_train)]
sanity_check_train = [dict_merge(x, {'y_est': y_est_train[idx]}) for idx,x in enumerate(sanity_check_train)]

sanity_check_test = v.inverse_transform(X_test)
sanity_check_test = [dict_merge(x, {'y': y_test[idx]}) for idx,x in enumerate(sanity_check_test)]
sanity_check_test = [dict_merge(x, {'y_est': y_est[idx]}) for idx,x in enumerate(sanity_check_test)]

# merge in estimates and movie info
item_data = pd.read_csv(data_path+'u.item', sep="|", header=None, names=['movie_id','movie_title','release_date','video_release_date','imdb_url','unknown','Action','Adventure','Animation','Children\'s','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'], encoding = "ISO-8859-1")
item_data['movie_id'] = item_data['movie_id'].astype(str)
item_data = item_data[['movie_id','movie_title']]

sanity_check_test = pd.DataFrame(sanity_check_test)
sanity_check_train = pd.DataFrame(sanity_check_train)

# for a given user, get top three historical movies against top three scored new movies
import time
for spot_id in range(1,100):
    time.sleep(5)
    top_ten_hist = sanity_check_train.loc[sanity_check_train['user_id']==spot_id,].sort_values(by=['y'], ascending=False)[0:10]
    top_ten_hist['movie_id'] = top_ten_hist.filter(regex='movie_id').astype(int).astype(str)
    top_ten_hist = top_ten_hist.merge(item_data, how='left', on='movie_id')
    top_five_pred = sanity_check_test.loc[sanity_check_train['user_id']==spot_id,].sort_values(by=['y_est'], ascending=False)[0:10]
    top_five_pred['movie_id'] = top_five_pred.filter(regex='movie_id').astype(int).astype(str)
    top_five_pred = top_five_pred.merge(item_data, how='left', on='movie_id')
    print(top_ten_hist[['movie_title','y', 'y_est']])
    print(top_five_pred[['movie_title','y','y_est']])

