import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# from plotly import graph_objs as go
from prophet.plot import plot_plotly

from prophet.serialize import  model_from_json
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import folium
from folium.plugins import Search, MarkerCluster
from streamlit_folium import folium_static
import pandas as pd
import geopandas as gpd
import smtplib
import datetime
from shapely.geometry import Point



st.set_page_config(page_title="Team cl_AI_mate", page_icon=":tada:", layout="wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def heat_index(df):
    df['datetime'] =  pd.to_datetime(df['datetime'], format='%Y%m%d %H:%M:%S')
    T=(df['temp']*9/5)+32  
    df['temp']=T
    R=df['humidity']
    hi = -42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 6.83783*(10**-3)*(T*T) - 5.481717*(10**-2)*R*R + 1.22874*(10**-3)*T*T*R + 8.5282*(10**-4)*T*R*R - 1.99*(10**-6)*T*T*R*R
    df['heat_index'] = hi
    return df


local_css("style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")


st.sidebar.header('Team cl_AI_mate')

st.sidebar.subheader('What you want to Predict?')
time_hist_color = st.sidebar.selectbox('Choose:', ('AQI', 'Heat wave')) 

st.sidebar.subheader('Choose a city:')
donut_theta = st.sidebar.selectbox('Select data', ('Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal'))

st.sidebar.subheader('Line chart parameters')
plot_data = st.sidebar.multiselect('Select data', ['temperature', 'humidity'], ['temperature', 'humidity'])
plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with ❤️ by [Team cl_AI_mate](https://github.com/Shivansh1203/Team-Cl_AI_mate).
''')
# ---- HEADER SECTION ----
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Team cl_AI_mate")
        st.title("Heatwave Prediction")
        st.write(
            "Telangana Tier-2 cities - Alidabad, Nizamabad, Karimnagar, Khammam and Warangal."
        )
        st.write("[Learn More >](https://github.com/Shivansh1203/Team-Cl_AI_mate)")

    with right_column:
        image = Image.open('images/hw2.jpg')
        st.image(image)
       

# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Our Approach")
        st.write("##")
        st.write(
            """
            The problem statemeint asks you to build a solution to predict two environmental factors in the Tier-2 cities of the Indian state of Telangana: 


            1. Heat Wave Occurrences: Heat waves are prolonged periods of excessively high temperatures, which can have severe impacts on public health and local ecosystems. The task is to develop a solution that can predict when heat waves will occur in the Tier-2 cities of Telangana, to make people aware of the future occurrence of the Heat wave. 

 

            2. Air Quality Index (AQI): AQI is a measure of the air quality in a given location. It takes into account various pollutants in the air and provides a single numerical value that represents the overall air quality. The goal is to predict the AQI in the Tier-2 cities of Telangana to help residents and local authorities make informed decisions about air quality and health. 

  
            The solution should be able to accurately predict both heat wave occurrences and AQI for the time frame January 2023 - December 2023 on a monthly basis, which can help to mitigate their impacts on public health and the environment. 
            """
        )

    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")
      

# ---- PROJECTS ----
with st.container():
 
 st.write("---")
  
    
 st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Our Model")

cities = ('Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal')
selected_city = st.selectbox('Select a city for prediction', cities)

@st.cache(allow_output_mutation=True)  #if running on vscode write only @st.cache_data
def load_prediction(city):
  path="winner/winner_{}_prediction.csv".format(city)
  df = pd.read_csv(path)
  return df
def load_model(city):
  path="winner/winner_{}_model.json".format(city)
  with open(path, 'r') as fin:
    m = model_from_json(fin.read())  # Load model
  return m

with st.spinner('Loading Model Into Memory....'):
  m= load_model(donut_theta)

forecast = load_prediction(donut_theta)


st.subheader('Predicted Data')
st.write(forecast.tail())

st.header("Graph")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)




with st.container():
    st.write("---")
    st.subheader("Choose a date")
    d = st.date_input(
    "",
    datetime.date(2019, 7, 6))
    st.header("Map")
    # Load the CSV file into a pandas DataFrame
    df_ad = pd.read_csv('content/Adilabad.csv')
    df_ka = pd.read_csv('content/Karimnagar.csv')
    df_kh = pd.read_csv('content/Khammam.csv')
    df_ni = pd.read_csv('content/Nizamabad.csv')
    df_wa = pd.read_csv('content/Adilabad.csv')

    # Convert the datetime column to a pandas datetime format
    df_ad['datetime'] = pd.to_datetime(df_ad['datetime'])
    df_ka['datetime'] = pd.to_datetime(df_ka['datetime'])
    df_kh['datetime'] = pd.to_datetime(df_kh['datetime'])
    df_ni['datetime'] = pd.to_datetime(df_ni['datetime'])
    df_wa['datetime'] = pd.to_datetime(df_wa['datetime'])

    df_ka.set_index('datetime', inplace=True)
    df_ad.set_index('datetime', inplace=True)
    df_kh.set_index('datetime', inplace=True)
    df_ni.set_index('datetime', inplace=True)
    df_wa.set_index('datetime', inplace=True)

    df_ad = df_ad.resample('d').max()
    df_ka = df_ka.resample('d').max()
    df_kh = df_kh.resample('d').max()
    df_ni = df_ni.resample('d').max()
    df_wa = df_wa.resample('d').max()

    df_ad = df_ad.reset_index()
    df_ka = df_ka.reset_index()
    df_kh = df_kh.reset_index()
    df_ni = df_ni.reset_index()
    df_wa = df_wa.reset_index()

    df_ad['date'] = df_ad['datetime'].dt.date
    df_ka['date'] = df_ka['datetime'].dt.date
    df_kh['date'] = df_kh['datetime'].dt.date
    df_ni['date'] = df_ni['datetime'].dt.date
    df_wa['date'] = df_wa['datetime'].dt.date
    # Set the datetime column as the DataFrame's index
    df_ka.set_index('date', inplace=True)
    df_ad.set_index('date', inplace=True)
    df_kh.set_index('date', inplace=True)
    df_ni.set_index('date', inplace=True)
    df_wa.set_index('date', inplace=True)

    temp_ad = df_ad.loc[d, 'temp']
    temp_ka = df_ka.loc[d, 'temp']
    temp_kh = df_kh.loc[d, 'temp']
    temp_ni = df_ni.loc[d, 'temp']
    temp_wa = df_wa.loc[d, 'temp']


    df_ad=heat_index(df_ad)
    df_ka=heat_index(df_ka)
    df_kh=heat_index(df_kh)
    df_ni=heat_index(df_ni)
    df_wa=heat_index(df_wa)
    # Select the temperature and heat index value for a particular date and store it in a variable

    heat_index_ad = df_ad.loc[d, 'heat_index']
    heat_index_ka = df_ka.loc[d, 'heat_index']
    heat_index_kh = df_kh.loc[d, 'heat_index']
    heat_index_ni = df_ni.loc[d, 'heat_index']
    heat_index_wa = df_wa.loc[d, 'heat_index']


    cities = {
        'city': ['Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal'],
        'heat index': [heat_index_ad, heat_index_ka, heat_index_kh, heat_index_ni, heat_index_wa],
        'Temperature(°C)': [temp_ad, temp_ka, temp_kh, temp_ni, temp_wa],
        'latitude': [19.6625054 , 18.6804717 , 18.4348833 , 17.2484683 , 17.9774221],
        'longitude': [78.4953182 , 78.0606503 , 79.0981286 , 80.006904 , 79.52881]
    }

    # Convert the city data to a GeoDataFrame
    geometry = [Point(xy) for xy in zip(cities['longitude'], cities['latitude'])]
    cities_gdf = gpd.GeoDataFrame(cities, geometry=geometry, crs='EPSG:4326')

    # Save the GeoDataFrame to a GeoJSON file
    cities_gdf.to_file('cities.geojson', driver='GeoJSON')



    # Load the city data
    cities = gpd.read_file("cities.geojson")

    # Create a folium map centered on the India
    m = folium.Map(location=[17.9774221, 79.52881], zoom_start=6)

    # Create a GeoJson layer for the city data
    geojson = folium.GeoJson(
        cities,
        name='City Data',
        tooltip=folium.GeoJsonTooltip(
            fields=['city', 'heat index', 'Temperature(°C)'],
            aliases=['City', 'heat index', 'Temperature(°C)'],
            localize=True
        )
    ).add_to(m)

    # Add a search bar to the map
    search = Search(
        layer=geojson,
        geom_type='Point',
        placeholder='Search for a city',
        collapsed=False,
        search_label='city'
    ).add_to(m)

    folium_static(m)


# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Get In Touch With Us")
    st.write("##")

 
    def send_email(name, email, message):
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login("teamclAImate2023@gmail.com", "balzsvjxdmgtsuts")
        msg = f"Subject: New message from {name}\n\n{name} ({email}) sent the following message:\n\n{message}"
        server.sendmail("teamclAImate2023@gmail.com", "teamclAImate2023@gmail.com", msg)
        st.success("Thank you for contacting us.")
        
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")

    if st.button("Send"):
        send_email(name, email, message)

    
    st.markdown(
    """
    <style>
       

         /* Adjust the width of the form elements */
        .stTextInput {
            width: 50%;

        }
        
        .stTextArea {
            width: 20%;
        }

        /* Style the submit button */
        .stButton button {
            background-color: #45a049;
            color: #FFFFFF;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            width: 10%;
        }

        /* Style the success message */
        .stSuccess {
            color: #0072C6;
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)





