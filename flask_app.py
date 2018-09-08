# A very simple Flask Hello World app for you to get started with...

from flask import Flask, redirect, render_template, request, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import math
import time
import datetime
from pytz import timezone

import tensorflow as tf
import keras
from statsmodels.tsa.arima_model import ARIMA

from pandas import datetime
from pandas_datareader import data
from stockstats import StockDataFrame

app = Flask(__name__)
app.config["DEBUG"] = True

start = datetime(2016,3,19)
end = datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d")

def modify_df(df):
    df = StockDataFrame.retype(df)
    y = df['close']
    df['macd'] = df.get('macd')
    df['rsi_12'] = df.get('rsi_6')
    df['volume_delta'] = df.get('volume_delta')
    df['MA 20'] = y.rolling(20).mean()
    df['MA 50'] = y.rolling(50).mean()
    df['Daily Change'] = df['close']-df['open']
    df['Fluctuation'] = ((df['high']-df['low'])/df['low'])*100
    X = df.drop(['open', 'high', 'low', 'macdh', 'macds', 'close_-1_s', 'close_-1_d', 'rs_6', 'rsi_6', 'close_12_ema', 'close_26_ema'], axis=1)
    X1 = X['20160103':]
    y = X1['close']
    y_max = np.max(np.array(y))
    y_min = np.min(np.array(y))
    return X1, y, y_max, y_min

from sklearn.preprocessing import MinMaxScaler
def normalize_df(df):
    normalize_data = MinMaxScaler()
    for i in range(0,df.shape[1]):
        df.iloc[:,i] = normalize_data.fit_transform(df.iloc[:,i].values.reshape(-1,1))
        df1 = df
    return df1

window = 10 #prediction time-lag window
def load_data_1(stock,y,window):
    raw_data = stock.as_matrix()
    length = raw_data.shape[0]
    indicators = raw_data.shape[1]
    prices = y.as_matrix()
    data = []

    for index in range(len(raw_data)-(window)+1):
        data.append(raw_data[index:index+window])

    data = np.array(data)
    valid_size = int(np.round(2/100*data.shape[0]))
    test_size = int(1)
    training_size = data.shape[0] - (valid_size + test_size)

    X_train = data[:training_size+valid_size,:-1]
    y_train = prices[window-1:training_size+window+valid_size-1]

    #validation for time series
    X_valid = data[training_size:training_size+valid_size,:-1]
    y_valid = prices[training_size+window-1:training_size+valid_size+window-1]
    raw_data = raw_data.reshape(1,length,indicators)

    X_test = np.zeros((1,window,8))
    for i in range(0,window-1):
        X_test[0,window-1-i,:] = raw_data[0,length-i-1,:]
    X_test = X_test[:,1:,:]
    y_test = 0
    return [X_train, y_train, X_valid, y_valid, X_test, y_test]

#X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(X3,y_normal,window+1)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras import regularizers

def build_model(layers,neurons,d):
    #d = 0.3
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))
    #used small L2 regularization here because it makes k folds CV more consistent
    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))


    start = time.time()
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def model_predict(X_train, y_train, X_test,model,y_max,y_min):
    model.fit(X_train,y_train,batch_size=512,epochs=90,validation_split=0.1,verbose=1)
    y_predict = model.predict(X_test)
    y_predict_scaled = y_predict*(y_max-y_min)+y_min
    return y_predict_scaled

def shift_y(y):
    y_normal1 = y[5:]
    y_normal1.shape
    y_normal1 = np.array(y_normal1)
    y_normal2 = pd.DataFrame(np.append(y_normal1,[0,0,0,0,0]))
    return y_normal2

def load_data_5(stock,y,window):
    raw_data = stock.as_matrix()
    length = raw_data.shape[0]
    indicators = raw_data.shape[1]
    prices = y.as_matrix()
    data = []

    for index in range(len(raw_data)-(window)+1):
        data.append(raw_data[index:index+window])

    data = np.array(data)
    valid_size = int(np.round(1/100*data.shape[0]))
    test_size = int(1)
    training_size = data.shape[0] - (4 + test_size)

    X_train = data[:training_size,:-1]
    y_train = prices[window-1:training_size+window-1]

    X_valid = data[training_size+valid_size:,:-1]
    y_valid = prices[-1]

    raw_data = raw_data.reshape(1,length,indicators)

    X_test = np.zeros((1,window,8))
    for i in range(0,window-1):
        X_test[0,window-1-i,:] = raw_data[0,length-i-1,:]
    X_test = X_test[:,1:,:]
    y_test = 0
    return [X_train, y_train, X_valid, y_valid, X_test, y_test]


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]



@app.route('/')
def index():
    return render_template('main_page.html')

@app.route("/", methods=["GET","POST"])
def index_post():
    if request.method == "POST":
        text = request.form['contents']
        processed_text = text.upper()
        stock = processed_text.replace("\r\n","")
        df = data.DataReader(stock, 'iex', start, end)
        X1, y, y_max, y_min = modify_df(df)
        X2 = normalize_df(X1)
        y_normal = X2['close']
        X_normal = X2.drop(['close'], axis = 1)
        [X_train1, y_train1, X_valid1, y_valid1, X_test1, y_test1] = load_data_1(X_normal,y_normal,window+1)
        model1 = build_model([X_normal.shape[1],window,1],[256,256,32,1],0.3)
        y_predict_scaled_1 = model_predict(X_train1, y_train1, X_test1,model1,y_max,y_min)
        print(y_predict_scaled_1)
        y_normal2 = shift_y(y_normal)
        [X_train5, y_train5, X_valid5, y_valid5, X_test5, y_test5] = load_data_5(X_normal,y_normal2,window+1)
        model5 = build_model([X_normal.shape[1],window,1],[256,256,32,1],0.3)
        y_predict_scaled_5 = model_predict(X_train5, y_train5, X_test5,model5,y_max,y_min)
        print(y_predict_scaled_5)
        y_1 = y[-1]*(y_max-y_min)+y_min
        if y_predict_scaled_1 > y_1:
            rec1 = 'Buy'
        else:
            rec1 = 'Sell'

        if y_predict_scaled_5 > y_1:
            rec5 = 'Buy'
        else:
            rec5 = 'Sell'

        today_date = datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d")
        return render_template("price_predict.html", stock=stock, y_plus_1=round(np.float(y_predict_scaled_1),2),
                               y_plus_5=round(np.float(y_predict_scaled_5),2),
                               last_y=round(y_1,2),today_date=today_date, rec_1=rec1,
                               rec_5=rec5)
    else:
        return render_template("main_page.html")
