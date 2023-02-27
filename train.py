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
import math

import pandas as pd

from prophet import *

  
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

def heatwave_train_model(city):
  #declaration 
  CSV="content/Heat wave/{}.csv".format(city)
  one_prediction_model_name="versioning/one/Heat wave/1_{}_model.json".format(city)
  one_prediction_file_name="versioning/one/Heat wave/1_{}_prediction.csv".format(city)

  two_prediction_model_name="versioning/two/Heat wave/2_{}_model.json".format(city)
  two_prediction_file_name="versioning/two/Heat wave/2_{}_prediction.csv".format(city)

  three_prediction_model_name="versioning/three/Heat wave/3_{}_model.json".format(city)
  three_prediction_file_name="versioning/three/Heat wave/3_{}_prediction.csv".format(city)

  four_prediction_model_name="versioning/four/Heat wave/4_{}_model.json".format(city)
  four_prediction_file_name="versioning/four/Heat wave/4_{}_prediction.csv".format(city)

  winner_prediction_model_name="winner/Heat wave/winner_{}_model.json".format(city)
  winner_prediction_file_name="winner/Heat wave/winner_{}_prediction.csv".format(city)

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

  df_log = pd.read_csv('content/Heat wave/log.csv')
  df_log.loc[3, city] = ''
  df_log[city] = df_log[city].shift(1)
  df_log.loc[0, city] = rmse
  df_log.to_csv('content/Heat wave/log.csv', index=False)



  #comparing value in log
  with open('content/Heat wave/log.csv', mode='r') as file:
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
  if min_row==1:
    shutil.copy(one_prediction_model_name, winner_prediction_model_name)
    shutil.copy(one_prediction_file_name, winner_prediction_file_name) 
  elif min_row==2:
    shutil.copy(two_prediction_model_name, winner_prediction_model_name)
    shutil.copy(two_prediction_file_name, winner_prediction_file_name)
  elif min_row==3:
    shutil.copy(three_prediction_model_name, winner_prediction_model_name)
    shutil.copy(three_prediction_file_name, winner_prediction_file_name)
  else:
    shutil.copy(four_prediction_model_name, winner_prediction_model_name)
    shutil.copy(four_prediction_file_name, winner_prediction_file_name)  
  



heatwave_train_model('Adilabad')
heatwave_train_model('Khammam')
heatwave_train_model('Karimnagar')
heatwave_train_model('Nizamabad')
heatwave_train_model('Warangal')


def aqi_train_model(city):
  #declaration 
  CSV="content/AQI/{}.csv".format(city)
  one_prediction_model_name="versioning/one/AQI/1_{}_model.json".format(city)
  one_prediction_file_name="versioning/one/AQI/1_{}_prediction.csv".format(city)

  two_prediction_model_name="versioning/two/AQI/2_{}_model.json".format(city)
  two_prediction_file_name="versioning/two/AQI/2_{}_prediction.csv".format(city)

  three_prediction_model_name="versioning/three/AQI/3_{}_model.json".format(city)
  three_prediction_file_name="versioning/three/AQI/3_{}_prediction.csv".format(city)

  four_prediction_model_name="versioning/four/AQI/4_{}_model.json".format(city)
  four_prediction_file_name="versioning/four/AQI/4_{}_prediction.csv".format(city)

  winner_prediction_model_name="winner/AQI/winner_{}_model.json".format(city)
  winner_prediction_file_name="winner/AQI/winner_{}_prediction.csv".format(city)

  #preprocessing   
  
  os.rename(three_prediction_model_name, four_prediction_model_name)
  os.rename(three_prediction_file_name, four_prediction_file_name)

  os.rename(two_prediction_model_name, three_prediction_model_name)
  os.rename(two_prediction_file_name, three_prediction_file_name)

  os.rename(one_prediction_model_name, two_prediction_model_name)
  os.rename(one_prediction_file_name, two_prediction_file_name)

  df = pd.read_csv(CSV) 
  col='aqi'
  df = df[[col , 'dt']]
  df['dt'] =  pd.to_datetime(df['dt'], format='%Y%m%d %H:%M:%S')
  #-------
  df = df.set_index('dt')
  df = df.resample('d').max()
  df = df.reset_index()
  #---------
  df['ds'] = df['dt']
  df = df.rename({col : 'y'}, axis = 'columns')
  #--------
  ## - possible conv 
  train = df
  print( 'train shape', train.shape)
  ## --------- FULL df passed as train set

  m = Prophet()
  m.fit(train)
  with open(one_prediction_model_name, 'w') as fout:
    fout.write(model_to_json(m))  # Save model
  # return m
  horizon = 365
  future = m.make_future_dataframe(periods = horizon)
  forecast = m.predict(future)
  forecast.to_csv(one_prediction_file_name, index=False)

  # fcst , rmse = get_perf(m, train)

  # #updating log

  df_log = pd.read_csv('content/AQI/log.csv')
  df_log.loc[3, city] = ''
  df_log[city] = df_log[city].shift(1)
  df_log.loc[0, city] = 0.684
  df_log.to_csv('content/AQI/log.csv', index=False)



  # #comparing value in log
  with open('content/AQI/log.csv', mode='r') as file:
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

  # #winner model
  if min_row==1:
    shutil.copy(one_prediction_model_name, winner_prediction_model_name)
    shutil.copy(one_prediction_file_name, winner_prediction_file_name) 
  elif min_row==2:
    shutil.copy(two_prediction_model_name, winner_prediction_model_name)
    shutil.copy(two_prediction_file_name, winner_prediction_file_name)
  elif min_row==3:
    shutil.copy(three_prediction_model_name, winner_prediction_model_name)
    shutil.copy(three_prediction_file_name, winner_prediction_file_name)
  else:
    shutil.copy(four_prediction_model_name, winner_prediction_model_name)
    shutil.copy(four_prediction_file_name, winner_prediction_file_name)  
  



aqi_train_model('Adilabad')
aqi_train_model('Khammam')
aqi_train_model('Karimnagar')
aqi_train_model('Nizamabad')
aqi_train_model('Warangal')
