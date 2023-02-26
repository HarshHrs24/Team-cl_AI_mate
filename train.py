import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime as dt
import csv
import shutil
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.serialize import model_to_json, model_from_json

import pandas as pd

from prophet import *

def drop_schema(df):
  drop_list = ['precip','windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover' ,'visibility',
        'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'conditions',
        'icon', 'stations', 'name','feelslike', 'dew','precipprob', 'preciptype', 'snow', 'snowdepth']
  df = df.drop(drop_list , axis=1)
  return df

def resample_schema(df , period , type):
  df_id = df.set_index('datetime')
  if(type):
    df_id_sampled = df_id.resample(period).mean()
  else:
    df_id_sampled = df_id.resample(period).max()
  return df_id_sampled

def split(df , time_col ,split ,lower = None , upper = None):

  df['ds'] = df.index
  # - 
  df = df.rename({'heat_index' : 'y'}, axis = 'columns')
  split_dt = dt.datetime(split , 1 , 1 , 0 ,0)
  split_dt -= dt.timedelta(days=1)

  print("split at" , split_dt)
  if(lower == None):
    lower = df[time_col].iloc[0]
  if(upper == None):
    upper = df[time_col].iloc[-1]

  train = df[ (df[time_col] <= split_dt) ]
  test = df[(df[time_col] > split_dt) ]

  print(train.shape , test.shape)
  print(train.iloc[-1][time_col] , test.iloc[0][time_col])
  
  return train ,test

# from sklearn.model_selection import train_test_split
# def prophet_perp(df , pred_col):
#   # - 
#   df['ds'] = df.index
#   # - 
#   df = df.rename({pred_col : 'y'}, axis = 'columns')
#   # - 
#   train, test = train_test_split(df, test_size=0.2 , shuffle= False)

#   # print(train.shape , test.shape)
#   # print(train.iloc[-1][time_col] , test.iloc[0][time_col])
  
#   return train , test
  
def prophet_train(train ):
  m = Prophet(
      
  )
  m.fit(train)
  print(m.params)
  return m

def get_perf(m , train, horizon = 365 ):
  """
  Parameters
  ----------
  m: model
  train: full train df on which model if fitted

  Returns
  -------
  Tuple(mae , rmse)
  """
  
  fcst_df = m.make_future_dataframe(periods = horizon)
  fcst = m.predict(fcst_df)
    
  perf_df = fcst[ : -horizon]
  perf_df['y'] = train['y']

  score_mae = mean_absolute_error(perf_df['y'] , perf_df['yhat'])
  score_rmse = math.sqrt(mean_squared_error(perf_df['y'] , perf_df['yhat']))

  print(score_mae)
  print(score_rmse)
  return (fcst , score_rmse)

def train_m(train):
  m = Prophet()
  m.fit(train)
  return m

def train_model(city):
  #declaration 
  CSV="content/{}.csv".format(city)
  one_prediction_model_name="versioning/one/1_{}_model.json".format(city)
  one_prediction_file_name="versioning/one/1_{}_prediction.csv".format(city)

  two_prediction_model_name="versioning/two/2_{}_model.json".format(city)
  two_prediction_file_name="versioning/two/2_{}_prediction.csv".format(city)

  three_prediction_model_name="versioning/three/3_{}_model.json".format(city)
  three_prediction_file_name="versioning/three/3_{}_prediction.csv".format(city)

  four_prediction_model_name="versioning/four/4_{}_model.json".format(city)
  four_prediction_file_name="versioning/four/4_{}_prediction.csv".format(city)

  winner_prediction_model_name="winner/winner_{}_model.json".format(city)
  winner_prediction_file_name="winner/winner_{}_prediction.csv".format(city)

  #preprocessing   
  
  os.rename(three_prediction_model_name, four_prediction_model_name)
  os.rename(three_prediction_file_name, four_prediction_file_name)

  os.rename(two_prediction_model_name, three_prediction_model_name)
  os.rename(two_prediction_file_name, three_prediction_file_name)

  os.rename(one_prediction_model_name, two_prediction_model_name)
  os.rename(one_prediction_file_name, two_prediction_file_name)

  df = pd.read_csv(CSV)
  print(df.head())
  
  df['datetime'] =  pd.to_datetime(df['datetime'], format='%Y%m%d %H:%M:%S')
  T=(df['temp']*9/5)+32  
  df['temp']=T
  R=df['humidity']
  hi = -42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 6.83783*(10**-3)*(T*T) - 5.481717*(10**-2)*R*R + 1.22874*(10**-3)*T*T*R + 8.5282*(10**-4)*T*R*R - 1.99*(10**-6)*T*T*R*R
  df['heat_index'] = hi
  col='heat_index'
  df = df[[col , 'datetime']]
  #-------
  df = df.set_index('datetime')
  df = df.resample('d').max()
  df = df.reset_index()
  #---------
  df['ds'] = df['datetime']
  df = df.rename({col : 'y'}, axis = 'columns')

  ## -possible conv 
  train = df
  ## --------- FULL df passed as train set

  m = train_m(train)

  fcst , rmse = get_perf(m, train)

  with open(one_prediction_model_name, 'w') as fout:
    fout.write(model_to_json(m))  # Save model
  future = m.make_future_dataframe(periods = 365)
  forecast = m.predict(future)
  forecast.to_csv(one_prediction_file_name, index=False)

  #updating log

  df_log = pd.read_csv('content/log.csv')
  df_log.loc[3, city] = ''
  df_log[city] = df_log[city].shift(1)
  df_log.loc[0, city] = rmse
  df_log.to_csv('content/log.csv', index=False)



  #comparing value in log
  with open('content/log.csv', mode='r') as file:
      reader = csv.reader(file)
      header = next(reader)
      col_index = header.index(city)
      min = float('inf')
      min_row = None
      for i, row in enumerate(reader):
          value = float(row[col_index])
          if value < min:
              min = value
              min_row = i + 1  

  #winner model
  if min_adilabad_row==1:
    shutil.copy(one_prediction_model_name, winner_prediction_model_name)
    shutil.copy(one_prediction_file_name, winner_prediction_file_name) 
  elif min_adilabad_row==2:
    shutil.copy(two_prediction_model_name, winner_prediction_model_name)
    shutil.copy(two_prediction_file_name, winner_prediction_file_name)
  elif min_adilabad_row==3:
    shutil.copy(three_prediction_model_name, winner_prediction_model_name)
    shutil.copy(three_prediction_file_name, winner_prediction_file_name)
  else:
    shutil.copy(four_prediction_model_name, winner_prediction_model_name)
    shutil.copy(four_prediction_file_name, winner_prediction_file_name)  
  



train_model('Adilabad')
train_model('Khammam')
train_model('Karimnagar')
train_model('Nizamabad')
train_model('Warangal')

train_model('Adilabad')
train_model('Khammam')
train_model('Karimnagar')
train_model('Nizamabad')
train_model('Warangal')
