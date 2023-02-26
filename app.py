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

from shapely.geometry import Point


# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
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
def load_model(city):
  path='json/{}_model.json'.format(city)
  with open(path, 'r') as fin:
    m = model_from_json(fin.read())  # Load model
  return m

with st.spinner('Loading Model Into Memory....'):
  m= load_model(donut_theta)



future = m.make_future_dataframe(periods = 365)
forecast = m.predict(future)


st.subheader('Predicted Data')
st.write(forecast.tail())

st.header("Graph")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


# st.markdown("[Watch Video...](https://youtu.be/TXSOitGoINE)")
# with st.container():
#     image_column, text_column = st.columns((1, 2))
#     with image_column:
#         st.image(img_contact_form)
#     with text_column:
#         st.subheader("How To Add A Contact Form To Your Streamlit App")
#         st.write(
#             """
#             Want to add a contact form to your Streamlit website?
#             In this video, I'm going to show you how to implement a contact form in your Streamlit app using the free service ‘Form Submit’.
#             """
#         )
#         st.markdown("[Watch Video...](https://youtu.be/FOULV9Xij_8)")

# Map

# Define some sample city data
with st.container():
    st.write("---")
    st.header("Map")
    # cities = {
    #     'city': ['Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal'],
    #     'country': ['India', 'India', 'India', 'India', 'India'],
    #     'population': [883305, 8537673, 3979576, 2693976, 2345678],
    #     'latitude': [19.6625054 , 18.6804717 , 18.4348833 , 17.2484683 , 17.9774221],
    #     'longitude': [78.4953182 , 78.0606503 , 79.0981286 , 80.006904 , 79.52881]
    # }

    # # Convert the city data to a GeoDataFrame
    # geometry = [Point(xy) for xy in zip(cities['longitude'], cities['latitude'])]
    # cities_gdf = gpd.GeoDataFrame(cities, geometry=geometry, crs='EPSG:4326')

    # Save the GeoDataFrame to a GeoJSON file
    # cities_gdf.to_file('cities.geojson', driver='GeoJSON')



    # Load the city data
    cities = gpd.read_file("cities.geojson")

    # Create a folium map centered on the India
    m = folium.Map(location=[17.9774221, 79.52881], zoom_start=6)

    # Create a GeoJson layer for the city data
    choropleth = folium.Choropleth(
        geo_data='cities.geojson',
        columns=('State Name', 'State Total Reports Quarter'),
        key_on='feature.properties.D_N',
        line_opacity=0.8,
        highlight=True
    )
    choropleth.geojson.add_to(m)

    # Add a search bar to the map
    # search = Search(
    #     layer=geojson,
    #     geom_type='Polygon',
    #     placeholder='Search for a city',
    #     collapsed=False,
    #     search_label='city'
    # ).add_to(m)


    # Display the map
    st_map = st_folium(map, width=700, height=450)


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




