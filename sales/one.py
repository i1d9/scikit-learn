from __future__ import division
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta, date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#import Keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split


df_sales = pd.read_csv('train.csv')

# convert date field from string to datetime
df_sales['date'] = pd.to_datetime(df_sales['date'])

# represent month in date field as its first day
df_sales['date'] = df_sales['date'].dt.year.astype(
    'str') + '-' + df_sales['date'].dt.month.astype('str') + '-01'
# groupby date and sum the sales
df_sales['date'] = pd.to_datetime(df_sales['date'])
df_sales = df_sales.groupby('date').sales.sum().reset_index()


# create a new dataframe to model the difference
df_diff = df_sales.copy()  # add previous sales to the next row
# drop the null values and calculate the difference
df_diff['prev_sales'] = df_diff['sales'].shift(1)
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])

# create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['prev_sales'], axis=1)  # adding lags
for inc in range(1, 13):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(
        inc)  # drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)


# import MinMaxScaler and create a new dataframe for LSTM model
df_model = df_supervised.drop(
    ['sales', 'date'], axis=1)  # split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values

# apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)  # reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)


X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


# Create the model
model = Sequential()
model.add(LSTM(4, batch_input_shape=(
    1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=100,
          batch_size=1, verbose=1, shuffle=False)

# Make the prediction
y_pred = model.predict(X_test,batch_size=1)
#for multistep prediction, you need to replace X_test values with the predictions coming from t-1


#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    print(np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
    
#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)


#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_sales[-7:].date)
act_sales = list(df_sales[-7:].sales)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index+1]
    result_list.append(result_dict)
    
df_result = pd.DataFrame(result_list)
print(df_result)
#for multistep prediction, replace act_sales with the predicted sales