# importing necessary libraries
import requests
import os
import datetime as dt
import csv
import shutil
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.serialize import model_to_json, model_from_json
import math
import time
import pandas as pd
from prophet import *


def cord(city):
# Initialize latitude and longitude
   dir = {
    'Adilabad':(19.6625054 ,78.4953182),
    'Nizamabad':(18.6804717 ,78.0606503),
    'Karimnagar':(18.4348833 ,79.0981286),
    'Khammam':(17.2484683 ,80.006904),
    'Warangal':(17.9774221 ,79.52881)
    }
   return dir[city]

#----Evaluating KPI's-------
def get_perf(m , train, horizon = 365 ):  
  fcst_df = m.make_future_dataframe(periods = horizon)
  fcst = m.predict(fcst_df)
    
  perf_df = fcst[ : -horizon]
  perf_df['y'] = train['y']

  score_mae = mean_absolute_error(perf_df['y'] , perf_df['yhat'])
  score_rmse = math.sqrt(mean_squared_error(perf_df['y'] , perf_df['yhat']))

  print(score_mae)
  print(score_rmse)
  return score_rmse

#-----Weekly Dataset updation--------------
def weekly_update(city,model):
    api_key = "9ce9f54d7dd8da5535e273da3e9f1a72"
    lat, lon = cord(city)
    retrain_log_path="retrain/{}/{}_retrain_log.csv".format(model,city)
    df = pd.read_csv(retrain_log_path)
    new_value1 = int(time.time())
    print(new_value1)
    start = df['last updated date'].iloc[-1] + 60 * 60
    end=df['last updated date'].iloc[-1]
    print(df['last updated date'].iloc[-1]+ 7*24*60*60)
    while(new_value1>end):    #Updating will continue until it reaches the current time.
      end=end+ 7*24*60*60
    end=end- 7*24*60*60
    new_value2 = end
    df = df.append({'last updated date': new_value2 , 'retrain datetime' :new_value1}, ignore_index=True)
    df.to_csv(retrain_log_path, index=False)
    


    # requesting AQI data from API
    if model=='AQI':
          url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&appid={api_key}"
          response = requests.get(url)         
          w = pd.DataFrame()
          if(response.status_code==200):
            data = response.json()
            for i in data['list']:
                # Extract the values from each dictionary and append to the DataFrame
                w = w.append({'aqi': i['main']['aqi'] ,
                'CO' :i['components']['co'],
                  'no2' :i['components']['no2'],
                  'o3' :i['components']['o3'],
                    'so2' :i['components']['so2'],
                    'pm2_5' :i['components']['pm2_5'],
                      'nh3' :i['components']['nh3'],
                      'dt' :dt.datetime.fromtimestamp(i['dt'])},
                        ignore_index=True)
                print(city,model)
          else:
             print("sleep")
             time.sleep(5)
              
          data_path='versioning/two/{}/2_{}_data.csv'.format(model,city)
          df_new = pd.read_csv(data_path)
          k=pd.concat([df_new, w])

    # requesting weather data from API
    else: 
          w = pd.DataFrame()
          while(end>=start):
              url=f"http://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={start}&appid=9ce9f54d7dd8da5535e273da3e9f1a72"
              response = requests.get(url)
              if(response.status_code==200):
                data = response.json()
                for d in data['data']:
                    # Extract the values from each dictionary and append to the DataFrame
                    w = w.append({
                        'datetime': dt.datetime.fromtimestamp(d['dt']),
                        'temp': d['temp'],
                        'humidity': d['humidity'],
                        'windspeed': d['wind_speed'],
                        'cloudcover': d['clouds'],
                        'conditions': d['weather'][0]['description'],
                    }, ignore_index=True)
                start=start+60*60
                print(city,model)
              else:
                 print("sleep")
                 time.sleep(5)
                 start=start+60*60
          data_path='versioning/two/{}/2_{}_data.csv'.format(model,city)
          df_new = pd.read_csv(data_path)
          df_new['datetime'] =  pd.to_datetime(df_new['datetime'])
          w['temp']=w['temp']-273.15
          k=pd.concat([df_new, w])
    return k
def heatwave_train_model(city):
  #declaration 
  one_prediction_model_name="versioning/one/Heat wave/1_{}_model.json".format(city)
  one_prediction_file_name="versioning/one/Heat wave/1_{}_prediction.csv".format(city)
  one_data_file_name="versioning/one/Heat wave/1_{}_data.csv".format(city)
  


  two_prediction_model_name="versioning/two/Heat wave/2_{}_model.json".format(city)
  two_prediction_file_name="versioning/two/Heat wave/2_{}_prediction.csv".format(city)
  two_data_file_name="versioning/two/Heat wave/2_{}_data.csv".format(city)

  three_prediction_model_name="versioning/three/Heat wave/3_{}_model.json".format(city)
  three_prediction_file_name="versioning/three/Heat wave/3_{}_prediction.csv".format(city)
  three_data_file_name="versioning/three/Heat wave/3_{}_data.csv".format(city)

  four_prediction_model_name="versioning/four/Heat wave/4_{}_model.json".format(city)
  four_prediction_file_name="versioning/four/Heat wave/4_{}_prediction.csv".format(city)
  four_data_file_name="versioning/four/Heat wave/4_{}_data.csv".format(city)

  winner_prediction_model_name="winner/Heat wave/winner_{}_model.json".format(city)
  winner_prediction_file_name="winner/Heat wave/winner_{}_prediction.csv".format(city)

  

  #preprocessing     
  os.rename(three_prediction_model_name, four_prediction_model_name)
  os.rename(three_prediction_file_name, four_prediction_file_name)
  os.rename(three_data_file_name, four_data_file_name)

  os.rename(two_prediction_model_name, three_prediction_model_name)
  os.rename(two_prediction_file_name, three_prediction_file_name)
  os.rename(two_data_file_name, three_data_file_name)

  os.rename(one_prediction_model_name, two_prediction_model_name)
  os.rename(one_prediction_file_name, two_prediction_file_name)
  os.rename(one_data_file_name, two_data_file_name)
  
  #weekly update via api
  df = weekly_update(city,'Heat wave') #latest data
  df.to_csv(one_data_file_name, index=False)

  df['datetime'] =  pd.to_datetime(df['datetime'])
  col = 'temp'

  df = df[[col , 'datetime']]
  df['datetime'] =  pd.to_datetime(df['datetime'])
  #-------

  df = df.set_index('datetime')
  df = df.resample('d').max()
  df = df.reset_index()
  #---------

  df['ds'] = df['datetime']
  df = df.rename({col : 'y'}, axis = 'columns')


  m = Prophet(seasonality_mode='multiplicative',
                  weekly_seasonality=True,
                  daily_seasonality=False,
              )

  m.add_seasonality(name='yearly', period=365, fourier_order=20)

  m.fit(df)

  future = m.make_future_dataframe(periods=365)

  future['yearly'] = future['ds'].apply(lambda x: x.year - 1)
  rmse=get_perf(m , df)
  print('rmse',rmse)
  forecast = m.predict(future)
  with open(one_prediction_model_name, 'w') as fout:
    fout.write(model_to_json(m))  # Save model
  print("model saved")
  forecast.to_csv(one_prediction_file_name, index=False)
  print("prediction file saved")


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
  one_prediction_model_name="versioning/one/AQI/1_{}_model.json".format(city)
  one_prediction_file_name="versioning/one/AQI/1_{}_prediction.csv".format(city)
  one_data_file_name="versioning/one/AQI/1_{}_data.csv".format(city)

  two_prediction_model_name="versioning/two/AQI/2_{}_model.json".format(city)
  two_prediction_file_name="versioning/two/AQI/2_{}_prediction.csv".format(city)
  two_data_file_name="versioning/two/AQI/2_{}_data.csv".format(city)

  three_prediction_model_name="versioning/three/AQI/3_{}_model.json".format(city)
  three_prediction_file_name="versioning/three/AQI/3_{}_prediction.csv".format(city)
  three_data_file_name="versioning/three/AQI/3_{}_data.csv".format(city)

  four_prediction_model_name="versioning/four/AQI/4_{}_model.json".format(city)
  four_prediction_file_name="versioning/four/AQI/4_{}_prediction.csv".format(city)
  four_data_file_name="versioning/four/AQI/4_{}_data.csv".format(city)

  winner_prediction_model_name="winner/AQI/winner_{}_model.json".format(city)
  winner_prediction_file_name="winner/AQI/winner_{}_prediction.csv".format(city)

  #preprocessing   
  
  os.rename(three_prediction_model_name, four_prediction_model_name)
  os.rename(three_prediction_file_name, four_prediction_file_name)
  os.rename(three_data_file_name, four_data_file_name)

  os.rename(two_prediction_model_name, three_prediction_model_name)
  os.rename(two_prediction_file_name, three_prediction_file_name)
  os.rename(two_data_file_name, three_data_file_name)

  os.rename(one_prediction_model_name, two_prediction_model_name)
  os.rename(one_prediction_file_name, two_prediction_file_name)
  os.rename(one_data_file_name, two_data_file_name)

  #weekly update via api 
  df = weekly_update(city,'AQI')
  df.to_csv(one_data_file_name, index=False)
  col='aqi'
  df['dt'] =  pd.to_datetime(df['dt'])
  df = df.set_index('dt')
  #----------------
  df = df.resample('d').max()
  df=df.dropna(axis=0)
  #----------------
  df = df.rename({col : 'y'}, axis = 'columns')
  #----------------
  df = df.reset_index()
  df['ds'] = df['dt']
  df = df[['y' ,'ds']]
  df
  #----------------
  m = Prophet(
              seasonality_mode='multiplicative',
              weekly_seasonality=True,
              daily_seasonality=False
              )
  m.add_seasonality(
                    name='yearly',
                    period=365,
                    fourier_order=20
                    )
  m.fit(df)
  rmse=get_perf(m,df)
  print(rmse)
  future = m.make_future_dataframe(periods=365) # horizon

  future['yearly'] = future['ds'].apply(lambda x: x.year - 1) # addition 
  
  forecast = m.predict(future)
  with open(one_prediction_model_name, 'w') as fout:
    fout.write(model_to_json(m))  # Save model
  print("model saved")
  forecast.to_csv(one_prediction_file_name, index=False)
  print("prediction file saved")

  # #updating log
  df_log = pd.read_csv('content/AQI/log.csv')
  df_log.loc[3, city] = ''
  df_log[city] = df_log[city].shift(1)
  df_log.loc[0, city] = rmse
  df_log.to_csv('content/AQI/log.csv', index=False)

  #comparing value in log
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


