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
import base64
from streamlit_timeline import st_timeline



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
selected_model = st.sidebar.selectbox('Choose:', ('Heat wave', 'AQI')) 

# st.sidebar.subheader('Choose a city:')
# donut_theta = st.sidebar.selectbox('Select data', ('Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal'))

# st.sidebar.subheader('Line chart parameters')
# plot_data = st.sidebar.multiselect('Select data', ['temperature', 'humidity'], ['temperature', 'humidity'])
# plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with ❤️ by [Team cl_AI_mate](https://github.com/HarshHrs24/Team-cl_AI_mate).

''')
                    
# def embed_pdf(pdf_file):
#     with open(pdf_file, "rb") as f:
#         data = f.read()
#     b64 = base64.b64encode(data).decode("utf-8")
#     pdf_display = f'<embed src="data:application/pdf;base64,{b64}" width="300" height="600" type="application/pdf">'
#     return pdf_display
# pdf_display = embed_pdf("json/Solution Architecture(Team cl_AI_mate).pdf")

with open('Architecture.pdf', "rb") as f:
    data = f.read()
b64 = base64.b64encode(data).decode("utf-8")
pdf_display = f'<embed src="data:application/pdf;base64,{b64}" width="300" height="600" type="application/pdf">'
st.sidebar.markdown(pdf_display, unsafe_allow_html=True)


# ---- HEADER SECTION ----
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Team cl_AI_mate")
        name="{} Prediction".format(selected_model)
        st.title(name)
        st.write(
            "Telangana Tier-2 cities - Alidabad, Nizamabad, Karimnagar, Khammam and Warangal."
        )
        st.write("[Learn More >](https://github.com/HarshHrs24/Team-cl_AI_mate)")

    with right_column:
        image = Image.open('images/hw2.jpg')
        st.image(image)
       


      



def heatwave_prepare(df):
   df['datetime'] = pd.to_datetime(df['datetime'])
   df.set_index('datetime', inplace=True)
   df = df.resample('d').max()
   df = df.reset_index()
   df['date'] = df['datetime'].dt.date
   df.set_index('date', inplace=True)
   T=(df['temp']*9/5)+32  
   df['temp']=T
   R=df['humidity']
   hi = -42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 6.83783*(10**-3)*(T*T) - 5.481717*(10**-2)*R*R + 1.22874*(10**-3)*T*T*R + 8.5282*(10**-4)*T*R*R - 1.99*(10**-6)*T*T*R*R
   df['heat_index'] = hi
   return df
def aqi_prepare(df):
   df['dt'] = pd.to_datetime(df['dt'])
   df.set_index('dt', inplace=True)
   df = df.resample('d').max()
   df = df.reset_index()
   df['date'] = df['dt'].dt.date
   df.set_index('date', inplace=True)
   return df

# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Our Approach")
        st.write("##")
        st.write(
            """
            The problem statement asks you to build a solution to predict two environmental factors in the Tier-2 cities of the Indian state of Telangana: 


            1. Heat Wave Occurrences: Heat waves are prolonged periods of excessively high temperatures, which can have severe impacts on public health and local ecosystems. The task is to develop a solution that can predict when heat waves will occur in the Tier-2 cities of Telangana, to make people aware of the future occurrence of the Heat wave. 

 

            2. Air Quality Index (AQI): AQI is a measure of the air quality in a given location. It takes into account various pollutants in the air and provides a single numerical value that represents the overall air quality. The goal is to predict the AQI in the Tier-2 cities of Telangana to help residents and local authorities make informed decisions about air quality and health. 

  
            The solution should be able to accurately predict both heat wave occurrences and AQI for the time frame January 2023 - December 2023 on a monthly basis, which can help to mitigate their impacts on public health and the environment. 
            """
        )

    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")


st.write("---")


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Our Model")

cities = ('Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal')
selected_city = st.selectbox('Select a city for prediction', cities)


   
# ---- PROJECTS ----
with st.container():


    @st.cache(allow_output_mutation=True)  #if running on vscode write only @st.cache_data
    def load_prediction(selected_model,city):
        path="winner/{}/winner_{}_prediction.csv".format(selected_model,city)
        df = pd.read_csv(path)
        return df
    
    def load_model(selected_model,city):
        path="winner/{}/winner_{}_model.json".format(selected_model,city)
        with open(path, 'r') as fin:
            m = model_from_json(fin.read())  # Load model
        return m

    with st.spinner('Loading Model Into Memory....'):
        m= load_model(selected_model,selected_city)

    forecast = load_prediction(selected_model,selected_city)


    st.header("Graph")
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
# Heat wave timeline
st.header(" Major Heat wave occurrences in the year 2023")
items = [
    {"id": 1, "content": "2023-01-20", "start": "2023-03-01"},
    {"id": 2, "content": "2023-10-09", "start": "2023-04-09"},
    {"id": 3, "content": "2023-10-18", "start": "2023-05-18"},
    {"id": 4, "content": "2023-10-16", "start": "2023-06-16"},
    {"id": 5, "content": "2023-10-25", "start": "2023-07-25"},
    {"id": 6, "content": "2023-10-27", "start": "2023-08-27"},
]

timeline = st_timeline(items, groups=[], options={}, height="300px")
st.subheader("Selected item")
st.write(timeline)
# events = [
#     {
#         "start": "2022-01-01",
#         "title": "Event 1",
#         "description": "This is event 1"
#     },
#     {
#         "start": "2022-02-01",
#         "title": "Event 2",
#         "description": "This is event 2"
#     },
#     {
#         "start": "2022-03-01",
#         "title": "Event 3",
#         "description": "This is event 3"
#     }
# ]




# selected_event = st_timeline(events, height=500, start_zoom=1, end_zoom=5)
# st.write(f"You selected the event '{selected_event['title']}'")

# Define start and end dates
start_date = datetime.date(2023, 1, 1)
end_date = datetime.date(2023, 12, 31)


# Create date input
selected_date = st.date_input(
    "Choose a date for the year 2023",
    value=datetime.date(2023, 1, 1),
    min_value=start_date,
    max_value=end_date,
    key="date_input"
)

# Display selected date
st.write("You selected:", selected_date.strftime("%B %d, %Y"))


st.write("---")
st.subheader("Choose a date")

if selected_model=='Heat wave':
    min_date = datetime.date(2012, 1, 1)
    max_date = datetime.date(2022, 12, 1)
else:
    min_date = datetime.date(2020, 12, 2)
    max_date = datetime.date(2022, 12, 1)
d = st.date_input(
"",
datetime.date(2021, 7, 6),
min_value=min_date,
max_value=max_date)
st.header("Map")
with st.container():

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        path_ad='content/{}/Adilabad.csv'.format(selected_model)
        path_ka='content/{}/Karimnagar.csv'.format(selected_model)
        path_kh='content/{}/Khammam.csv'.format(selected_model)
        path_ni='content/{}/Nizamabad.csv'.format(selected_model)
        path_wa='content/{}/Warangal.csv'.format(selected_model)
        df_ad = pd.read_csv(path_ad)
        df_ka = pd.read_csv(path_ka)
        df_kh = pd.read_csv(path_kh)
        df_ni = pd.read_csv(path_ni)
        df_wa = pd.read_csv(path_wa)

        if selected_model=='Heat wave':
            df_ad = heatwave_prepare(df_ad)
            df_ka = heatwave_prepare(df_ka)
            df_kh = heatwave_prepare(df_kh)
            df_ni = heatwave_prepare(df_ni)
            df_wa = heatwave_prepare(df_wa)

            temp_ad = df_ad.loc[d, 'temp']
            temp_ka = df_ka.loc[d, 'temp']
            temp_kh = df_kh.loc[d, 'temp']
            temp_ni = df_ni.loc[d, 'temp']
            temp_wa = df_wa.loc[d, 'temp']
        # Select the temperature and heat index value for a particular date and store it in a variable

            heat_index_ad = df_ad.loc[d, 'heat_index']
            heat_index_ka = df_ka.loc[d, 'heat_index']
            heat_index_kh = df_kh.loc[d, 'heat_index']
            heat_index_ni = df_ni.loc[d, 'heat_index']
            heat_index_wa = df_wa.loc[d, 'heat_index']
            cities = {
                'city': ['Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal'],
                'Heat Index': [heat_index_ad, heat_index_ka, heat_index_kh, heat_index_ni, heat_index_wa],
                'Temperature(°F)': [temp_ad, temp_ka, temp_kh, temp_ni, temp_wa],
                'latitude': [19.6625054 , 18.6804717 , 18.4348833 , 17.2484683 , 17.9774221],
                'longitude': [78.4953182 , 78.0606503 , 79.0981286 , 80.006904 , 79.52881]
            }

            # Convert the city data to a GeoDataFrame
            geometry = [Point(xy) for xy in zip(cities['longitude'], cities['latitude'])]
            cities_gdf = gpd.GeoDataFrame(cities, geometry=geometry, crs='EPSG:4326')

            # Save the GeoDataFrame to a GeoJSON file
            cities_gdf.to_file('heatwave_cities.geojson', driver='GeoJSON')



            # Load the city data
            cities = gpd.read_file("heatwave_cities.geojson")

            # Create a folium map centered on the India
            m = folium.Map(location=[17.9774221, 79.52881], zoom_start=6)

            # Create a GeoJson layer for the city data
            geojson = folium.GeoJson(
                cities,
                name='City Data',
                tooltip=folium.GeoJsonTooltip(
                    fields=['city', 'Heat Index', 'Temperature(°F)'],
                    aliases=['City', 'Heat Index', 'Temperature(°F)'],
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

            folium_static(m, width=500, height=500)
        else:
            df_ad = aqi_prepare(df_ad)
            df_ka = aqi_prepare(df_ka)
            df_kh = aqi_prepare(df_kh)
            df_ni = aqi_prepare(df_ni)
            df_wa = aqi_prepare(df_wa)

            aqi_ad = df_ad.loc[d, 'aqi']
            aqi_ka = df_ka.loc[d, 'aqi']
            aqi_kh = df_kh.loc[d, 'aqi']
            aqi_ni = df_ni.loc[d, 'aqi']
            aqi_wa = df_wa.loc[d, 'aqi']
        # Select the temperature and heat index value for a particular date and store it in a variable

            cities = {
                'city': ['Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal'],
                'AQI': [aqi_ad, aqi_ka, aqi_kh, aqi_ni, aqi_wa],
                'latitude': [19.6625054 , 18.6804717 , 18.4348833 , 17.2484683 , 17.9774221],
                'longitude': [78.4953182 , 78.0606503 , 79.0981286 , 80.006904 , 79.52881]
            }

            # Convert the city data to a GeoDataFrame
            geometry = [Point(xy) for xy in zip(cities['longitude'], cities['latitude'])]
            cities_gdf = gpd.GeoDataFrame(cities, geometry=geometry, crs='EPSG:4326')

            # Save the GeoDataFrame to a GeoJSON file
            cities_gdf.to_file('aqi_cities.geojson', driver='GeoJSON')



            # Load the city data
            cities = gpd.read_file("aqi_cities.geojson")

            # Create a folium map centered on the India
            m = folium.Map(location=[17.9774221, 79.52881], zoom_start=6)

            # Create a GeoJson layer for the city data
            geojson = folium.GeoJson(
                cities,
                name='City Data',
                tooltip=folium.GeoJsonTooltip(
                    fields=['city', 'AQI'],
                    aliases=['City', 'AQI'],
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

            folium_static(m, width=500, height=500)
        
    with middle_column:
        st.write("                 ")


    with right_column:
        image = Image.open('images/hic1.jpeg')
        image2 = Image.open('images/hic2.jpeg')
        st.image(image)
        st.image(image2)









# Description
def info(title, text):
    with  st.expander(f"{title}"):
        st.write(text)

st.write("---")
info("Description", "It will give the details of relevant parameters in reference to respective date and selected city.")

with st.container():
    path="content/{}/{}.csv".format(selected_model,selected_city)

    if selected_model=='Heat wave':

        df = pd.read_csv(path)
        df = heatwave_prepare(df)



        left_column, middle_column1, middle_column, right_column, middle_column2 = st.columns(5)
        with left_column:
            st.write("<p style='color: #00C957; font-size: 20px;'>Temperature(°F) : </p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>Humidity : </p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>Preciptation : </p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>Wind speed : </p>", unsafe_allow_html=True)
        with middle_column1:
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(df.loc[d, 'temp']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(df.loc[d, 'humidity']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(df.loc[d, 'precip']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(df.loc[d, 'windspeed']), unsafe_allow_html=True) 
        with middle_column:   
            st.write("                 ")   
        with right_column:
            st.write("<p style='color: #00C957; font-size: 20px;'>Cloud cover : </p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>Solar Radiation : </p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>UV Index : </p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>Condition : </p>", unsafe_allow_html=True)
        with middle_column2:
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(df.loc[d, 'cloudcover']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(df.loc[d, 'solarradiation']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(df.loc[d, 'uvindex']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(df.loc[d, 'conditions']), unsafe_allow_html=True)
    else:
        df = pd.read_csv(path)
        df = aqi_prepare(df)
        left_column, middle_column1, right_column, middle_column2 = st.columns(4)
        with left_column:
            st.write("<p style='color: #00C957; font-size: 20px;'>Carbon monoxide</p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>Nitrogen monoxide</p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>Nitrogen dioxide</p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>Ozone</p>", unsafe_allow_html=True)
        with middle_column1:
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(df.loc[d, 'co']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(df.loc[d, 'no']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(df.loc[d, 'no2']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(df.loc[d, 'o3']), unsafe_allow_html=True)   
        with right_column:
            st.write("<p style='color: #00C957; font-size: 20px;'>Sulphure dioxide</p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>Fine particles matter</p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>Coarse particle matter</p>", unsafe_allow_html=True)
            st.write("<p style='color: #00C957; font-size: 20px;'>Ammonia</p>", unsafe_allow_html=True)
        with middle_column2:
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(df.loc[d, 'so2']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(df.loc[d, 'pm2_5']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(df.loc[d, 'pm10']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(df.loc[d, 'nh3']), unsafe_allow_html=True)
    





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





