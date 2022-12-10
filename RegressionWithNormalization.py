import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import math

print(time.localtime())

# load data
all_data = np.genfromtxt('liverdata-min-50k-120822-ptimeyear.csv', delimiter=',', skip_header=1, filling_values=0)
#print('all data shape ', {all_data.shape})

# separate data into id col, feat cols, tgt col
id_data = all_data[:, 0:1]
#print('id col shape ', {id_data.shape})

#feat_data = all_data[:, 1:29]
feat_data = all_data[:, 1:33]
#print('feature data shape ', {feat_data.shape})

tgt_data = all_data[:, 34:]
#print('target col shape ', {tgt_data.shape})

# split data into training and test data
feat_train_data, feat_test_data, tgt_train_data, tgt_test_data = train_test_split(feat_data, tgt_data, test_size=0.4, random_state=0)
#print('feat train data shape ', {feat_train_data.shape})
#print('target train col shape ', {tgt_train_data.shape})
#print('feat test data shape ', {feat_test_data.shape})
#print('target test col shape ', {tgt_test_data.shape})

# normalize the features for training and test data
#norm = preprocessing.Normalizer
#norm_feat_train_data = norm.fit_transform(feat_train_data, axis=0, norm='l2')
#norm_feat_test_data = norm.transform(feat_test_data, axis=0, norm='l2')
#norm_feat_all_data = norm.transform(feat_data, axis=0, norm='l2')

# convert target columns to arrays
tgt_train_array = tgt_train_data.ravel()
#print('target train array shape ', {tgt_train_array.shape})

tgt_test_array = tgt_test_data.ravel()
#print('target test array shape ', {tgt_test_array.shape})

tgt_array = tgt_data.ravel()
#print('target array shape ', {tgt_array.shape})

# pipeline
sgd_reg = make_pipeline(StandardScaler(), SGDRegressor())
#sgd_reg.fit(norm_feat_train_data, tgt_train_array)
#pred_train_data = sgd_reg.predict(norm_feat_train_data)
#train_score = sgd_reg.score(norm_feat_train_data, tgt_train_array)

sgd_reg.fit(feat_train_data, tgt_train_array)
pred_train_data = sgd_reg.predict(feat_train_data)
#train_score = sgd_reg.score(feat_train_data, tgt_train_array)
#print('test data score ', train_score)

#test_score = sgd_reg.score(norm_feat_test_data, tgt_test_array)
test_score = sgd_reg.score(feat_test_data, tgt_test_array)
print('test data score ', test_score)

#pred_test_data = sgd_reg.predict(norm_feat_test_data)
pred_test_data = sgd_reg.predict(feat_test_data)
#print('test data, regression prediction shape', {pred_test_data.shape})

pred_test_data_matrix = np.asmatrix(pred_test_data).transpose()
#print('test data, regression prediction matrix shape', {pred_test_data_matrix.shape})

# replace -ve values to zero
pred_test_data_matrix[pred_test_data_matrix < 0] = 0

#diff_matrix = pred_test_data_matrix - tgt_test_data

MSE = mean_squared_error(tgt_test_data, pred_test_data_matrix)
print('test data MSE ', {MSE})

RMSE = math.sqrt(MSE)
print('test data RMSE  ', {RMSE})

#sgd_test_data_pred = np.concatenate((feat_test_data, tgt_test_data, pred_test_data_matrix, diff_matrix), axis=1)
#print('regression pred vs test pred ', {sgd_test_data_pred.shape})

#np.savetxt('liverdata-sgdpred-test-29-120722-2.csv', sgd_test_data_pred, delimiter=',', fmt='%10.5f')

#pred_all_data = sgd_reg.predict(norm_feat_all_data)
pred_all_data = sgd_reg.predict(feat_data)
#print('all data, regression prediction shape', {pred_all_data.shape})

#score = sgd_reg.score(norm_feat_all_data, tgt_array)
score = sgd_reg.score(feat_data, tgt_array)
print('all data score ', score)

pred_all_data_matrix = np.asmatrix(pred_all_data).transpose()
#print('all data, regression prediction matrix shape', {pred_all_data_matrix.shape})

pred_all_data_matrix[pred_all_data_matrix < 0] = 0

MSE = mean_squared_error(tgt_data, pred_all_data_matrix)
print('all data MSE ', {MSE})

RMSE = math.sqrt(MSE)
print('all data RMSE  ', {RMSE})

all_diff_matrix = pred_all_data_matrix - tgt_data

sgd_all_data_pred = np.concatenate((id_data, feat_data, tgt_data, pred_all_data_matrix, all_diff_matrix), axis=1)
#print('regression pred vs test pred ', {sgd_all_data_pred.shape})

#np.savetxt('liverdata-sgdpred-50k-33-120822-1.csv', sgd_all_data_pred, delimiter=',', fmt='%10.5f')

sgd_all_data_pred2 = np.concatenate((all_data, pred_all_data_matrix, all_diff_matrix), axis=1)
np.savetxt('liverdata-sgdpred-50k-ptimeyr-120822-1.csv', sgd_all_data_pred2, delimiter=',', fmt='%10.5f')
