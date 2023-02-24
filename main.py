import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np

import pandas as pd
import tensorflow as tf
import streamlit as st

# from plotly import graph_objs as go
from prophet.plot import plot_plotly

from prophet.serialize import  model_from_json



st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Sample websapp")

cities = ('Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal')
selected_city = st.selectbox('Select dataset for prediction', cities)

@st.cache(allow_output_mutation=True)
def load_model(city):
  # path='json/{}_model.json'.format(city)
  path='json/{}_model.json'.format(city)
  with open(path, 'r') as fin:
    m = model_from_json(fin.read())  # Load model
  return m

with st.spinner('Loading Model Into Memory....'):
  m= load_model(selected_city)



future = m.make_future_dataframe(periods = 365)
forecast = m.predict(future)


st.subheader('predicted data')
st.write(forecast.tail())

st.subheader('graph')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
