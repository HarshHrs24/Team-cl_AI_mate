import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime as dt

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

def train_model(city):
  CSV="content/{}.csv".format(city)
  df = pd.read_csv(CSV)
  print(df.head())
  df['datetime'] =  pd.to_datetime(df['datetime'], format='%Y%m%d %H:%M:%S')
  df = drop_schema(df) #function call
  T=(df['temp']*9/5)+32  
  df['temp']=T
  R=df['humidity']
  hi = -42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 6.83783*(10**-3)*(T*T) - 5.481717*(10**-2)*R*R + 1.22874*(10**-3)*T*T*R + 8.5282*(10**-4)*T*R*R - 1.99*(10**-6)*T*T*R*R
  df['heat_index'] = hi
  df = df.drop(['temp','humidity'] , axis=1)
  df_id_d = resample_schema(df , 'D' , 1) #function call
  df_id_d = df_id_d.loc[:'2022']
  train, test = split(df_id_d , 'ds', 2021) #function call
#   train , test = prophet_perp(df_id_d , 'heat_index') #function call
  m = prophet_train(train) #function call
  path="json/{}_model.json".format(city)
  modelname="{}_model.json".format(city)
  with open(path, 'w') as fout:
    fout.write(model_to_json(m))  # Save model
  with open(path, 'r') as fin:
    m1 = model_from_json(fin.read())  # Load model
    print(m1)
    print('[set target path 1]')
    target_path_1 = os.path.join(os.path.dirname(__file__), modelname)
    print(target_path_1)

train_model('Adilabad')
train_model('Khammam')
train_model('Karimnagar')
train_model('Nizamabad')
train_model('Warangal')
