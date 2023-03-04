# importing necessary libraries
import plotly.graph_objects as go
from streamlit_timeline import st_timeline
import base64
from shapely.geometry import Point
import datetime
import smtplib
import geopandas as gpd
import pandas as pd
from streamlit_folium import folium_static
from folium.plugins import Search
import folium
from PIL import Image
from streamlit_lottie import st_lottie
import streamlit as st
import requests
from prophet.serialize import model_from_json
from prophet.plot import plot_plotly
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# defining necessary functions

#to load necessary assests
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#dataset preperation heatwave
def heatwave_prepare(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.resample('d').max()
    df = df.reset_index()
    df['date'] = df['datetime'].dt.date
    df.set_index('date', inplace=True)
    T = (df['temp']*9/5)+32
    df['temp'] = T
    R = df['humidity']
    #Calculating Heat index using heat index chart formula
    hi = -42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 6.83783 * \
        (10**-3)*(T*T) - 5.481717*(10**-2)*R*R + 1.22874*(10**-3) * \
        T*T*R + 8.5282*(10**-4)*T*R*R - 1.99*(10**-6)*T*T*R*R
    df['heat_index'] = hi
    df['occurence of heat wave'] = df["temp"].apply(
        lambda x: "yes" if x > 128 else "no")
    return df


def conv(x):
    return round(x)

#dataset preperation for timeleine 
def timeline_prepare(df, model):
    if model == "Heat wave":
        df['occurence of heat wave'] = df["yhat_upper"].apply(
            lambda x: "yes" if x >= 44.9 else "no")
        df = df.iloc[4017:]

    else:
        df['yhat'] = df['yhat'].apply(conv)
        df['Extreme AQI events'] = df["yhat"].apply(
            lambda x: "yes" if x > 4 else "no")
    return df

#dataset preperation for AQI
def aqi_prepare(df):
    df['dt'] = pd.to_datetime(df['dt'])
    df.set_index('dt', inplace=True)
    df = df.resample('d').max()
    df = df.reset_index()
    df['date'] = df['dt'].dt.date
    df.set_index('date', inplace=True)
    return df

#Graph visualizations
def line_plot_plotly(m, forecast, mode, model):
    past = m.history['y']
    future = forecast['yhat']
    if model == 'AQI':
        future = future.apply(conv)
    timeline = forecast['ds']

    trace1 = go.Scatter(
        x=timeline,
        y=past,
        mode=mode,
        name='Actual',
        line=dict(color='#777777')
    )
    trace2 = go.Scatter(
        x=timeline,
        y=future,
        mode=mode,
        name='Predicted',
        line=dict(color='#FF7F50')
    )

    data = [trace1, trace2]

    layout = go.Layout(
        title='Actual vs. Predicted Values',
        xaxis=dict(title='Date', rangeslider=dict(visible=True),
                   rangeselector=dict(
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )),
        yaxis=dict(title='Value'),
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)

    return fig

# to load next year prediction
def load_prediction(selected_model, city):
    path = "winner/{}/winner_{}_prediction.csv".format(selected_model, city)
    print(path)
    df = pd.read_csv(path)
    return df

# to load model
def load_model(selected_model, city):
    path = "winner/{}/winner_{}_model.json".format(selected_model, city)
    with open(path, 'r') as fin:
        m = model_from_json(fin.read())  # Load model
    return m

# Description
def info(title, text):
    with st.expander(f"{title}"):
        st.write(text)

#mail
def send_email(name, email, message):
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login("teamclAImate2023@gmail.com", "balzsvjxdmgtsuts")
    msg = f"Subject: New message from {name}\n\n{name} ({email}) sent the following message:\n\n{message}"
    server.sendmail("teamclAImate2023@gmail.com",
                    "teamclAImate2023@gmail.com", msg)
    st.success("Thank you for contacting us.")


# ---- LOAD ASSETS ----

lottie_coding_1 = load_lottieurl(
    "https://assets5.lottiefiles.com/packages/lf20_2cwDXD.json")

lottie_coding_2 = load_lottieurl(
    "https://assets9.lottiefiles.com/packages/lf20_dXP5CGL9ik.json")

# ---- Title SECTION ----
st.set_page_config(page_title="Team cl_AI_mate",
                   page_icon=":tada:", layout="wide")

# ---- CSS----
local_css("style/style.css")


st.sidebar.header('Team cl_AI_mate')

st.sidebar.subheader('What you want to Predict?')
selected_model = st.sidebar.selectbox('Choose:', ('Heat wave', 'AQI'))
st.sidebar.write('''

''')
cities = ('Adilabad', 'Nizamabad', 'Karimnagar', 'Khammam', 'Warangal')
selected_city = st.sidebar.selectbox('Select a city for prediction', cities)

image = Image.open('images/logo.png')
st.sidebar.image(image)


st.sidebar.markdown('''
---
Created with ❤️ by [Team cl_AI_mate](https://github.com/iamneo-production/00aa9422-7c04-4b7c-975b-6ed887ff7d95).

''')


# ---- HEADER SECTION ----
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Team cl_AI_mate")
        st.write("Stay ahead of the heat and breathe easy with Team cl_AI_mate")
        name = "{} Prediction".format(selected_model)
        st.title(name)
        st.write(
            "Telangana Tier-2 cities - Alidabad, Nizamabad, Karimnagar, Khammam and Warangal."
        )
        if selected_model == "Heat wave":
            st.write("[Exploratory Data Analysis(EDA)](https://colab.research.google.com/drive/1xH77_KLE3gpmTxGk9-X36Pj6lHee0iBc?usp=sharing#scrollTo=nybHfIsygGzp)")
            st.write("[Solution Architecture](https://www.craft.do/s/1eTduABsPuFIDX)")

        else:
            st.write("[Exploratory Data Analysis(EDA)](https://colab.research.google.com/drive/1WgV57xtbG05shrxy47Fw59oOmzTZ2yJv?usp=sharing)")
            st.write("[Solution Architecture](https://www.craft.do/s/1eTduABsPuFIDX)")

    with right_column:
        i = 'images/{}_hw2.jpg'.format(selected_model)
        image = Image.open(i)
        st.image(image)


# ---- Introduction ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Our Vision and Approach")
        st.write(
            """
            Welcome to Team cl_AI_mate a home of Heatwave and AQI Prediction Platform! We are here to help you prepare for extreme weather conditions and make informed decisions to protect yourself and your loved ones.

            Our platform offers a seamless user journey, starting with the homepage where you can select the criteria you want to predict and the city you are interested in. Once you make your selection, our platform provides you with a graphical representation of the selected criteria for the chosen city, giving you a quick overview of the situation.

            The platform also includes polar plots and maps for analyzing trends and a map feature for visualizing data for selected cities. Our proposed solution architecture is scalable, adaptable, cost-effective, and dynamic due to retraining and versioning, and CI/CD implementation. Our platform offers a smooth and interactive user experience, providing all necessary information and insights about heatwave and AQI prediction for selected cities.
            """
        )

    with right_column:
        if selected_model == 'Heat wave':
            st_lottie(lottie_coding_1, height=300, key="coding")
        else:
            st_lottie(lottie_coding_2, height=300, key="coding")


st.write("---")


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Our Model")
st.write("Select the desired criterias from the sidebar")

# ---- Forecast ----
with st.container():

    with st.spinner('Loading Model Into Memory....'):
        m = load_model(selected_model, selected_city)

    forecast = load_prediction(selected_model, selected_city)


path1 = "winner/{}/winner_{}_prediction.csv".format(
    selected_model, selected_city)

st.header("Graph")
if selected_model == 'Heat wave':
    info("Info", '''The Graph displays the forecasted values and their associated uncertainty intervals over time. 
    Shaded areas above and below the line represent the uncertainty interval.
    The blue line represents the forecast prediction.''')

    agree = st.checkbox('Line graph')

    if agree:
        fig1 = line_plot_plotly(m, forecast, 'lines', selected_model)

        fig1.update_layout(
            plot_bgcolor='#7FFFD4',  # set the background color
            paper_bgcolor='#F8F8F8',  # set the background color of the plot area
        )

    else:
        fig1 = plot_plotly(m, forecast)

        fig1.update_layout(
            plot_bgcolor='#7FFFD4',  # set the background color
            paper_bgcolor='#F8F8F8',  # set the background color of the plot area
        )
else:
    info("Info", '''The Graph displays the prediction and actual AQI Reading for the range of the full dataset and for year 2023
    The orange points shows the predicted value and the grey points shows the actual value of AQI.''')
    fig1 = line_plot_plotly(m, forecast, 'markers', selected_model)

    fig1.update_layout(
        plot_bgcolor='#7FFFD4',  # set the background color
        paper_bgcolor='#F8F8F8',  # set the background color of the plot area
    )


st.plotly_chart(fig1)

# ---- Timeline ----

if selected_model == 'Heat wave':
    path = "winner/{}/winner_{}_prediction.csv".format(
        selected_model, selected_city)

    df = pd.read_csv(path)
    df = timeline_prepare(df, selected_model)

    df = df[df["occurence of heat wave"] == "yes"]
    # Convert the dataframe to a list of dictionaries
    items = []
    i = 1
    for index, row in df.iterrows():
        yhat_upper = str(row["yhat_upper"])
        yhat_lower = str(row["yhat_lower"])
        content = "On {}, {} is expected to experience a maximum temperature of {} and a minimum temperature of {}.".format(
            str(row["ds"]), selected_city, yhat_upper, yhat_lower)
        item = {"id": i, "content": "⚠",
                "message": content, "start": str(row["ds"])}
        items.append(item)
        i = i+1

    timeine_title = "Major Heat wave occurrences in the year 2023"
    st.header(timeine_title)
    info("Info", "The timeline highlights the major events in the year 2023 regarding the occurrence of Heat waves.")

    options = {
        "min": "2023-01-01",
        "max": "2023-12-31"
    }

    timeline = st_timeline(items, groups=[], options=options, height="300px")
    st.subheader("Selected item")
    st.write(timeline)
else:
    path = "winner/{}/winner_{}_prediction.csv".format(
        selected_model, selected_city)

    df = pd.read_csv(path)
    df = timeline_prepare(df, selected_model)
    df = df[df["Extreme AQI events"] == "yes"]

    # Convert the dataframe to a list of dictionaries
    items = []
    i = 1
    for index, row in df.iterrows():
        yhat = str(row["yhat"])
        content = "The predicted AQI for {} on {} is {}".format(
            selected_city, str(row["ds"]), yhat)
        item = {"id": i, "content": "⚠",
                "message": content, "start": str(row["ds"])}
        items.append(item)
        i = i+1

    timeine_title = "Major events in the year 2023 regarding severe Air Quality conditions."
    st.header(timeine_title)

    info("Info", "The timeline highlights the major events in the year 2023 regarding severe Air Quality conditions")

    options = {
        "min": "2023-01-01",
        "max": "2023-12-31"
    }

    timeline = st_timeline(items, groups=[], options=options, height="300px")
    st.subheader("Selected item")
    st.write(timeline)

# ---- specific date in the year 2023 to get the details about temperatue and AQI ----
with st.container():
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
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
        forecast = load_prediction(selected_model, selected_city)
        d = selected_date.strftime("%Y-%m-%d")
        yhat = "{:.2f}".format(
            float(forecast.loc[forecast['ds'] == d, 'yhat']))
        yhat_upper = "{:.2f}".format(
            float(forecast.loc[forecast['ds'] == d, 'yhat_upper']))
        yhat_lower = "{:.2f}".format(
            float(forecast.loc[forecast['ds'] == d, 'yhat_lower']))
        if selected_model == 'Heat wave':
            prediction_year_info = "On {} the predicted temperature range for Adilabad is between {} and {}, with a most likely temperature of {}.".format(
                d, yhat_upper, yhat_lower, yhat)
        else:
            prediction_year_info = "The predicted AQI for {} on {} is {}".format(
                selected_city, d, yhat)
        st.write(prediction_year_info)

# Polar Plots
st.write("---")
st.header("Polar Graph")
with st.container():
    if selected_model == 'Heat wave':
        info("Info", "City-wise temperature and Humidity data on a polar graph with the distance from the centre of the graph representing the value at that point in time, each rotation accounts for a full year.")
        c1, c2 = st.columns(2)
        with c1:

            """### Temperature trend over the decade"""
            gif1 = "images/Heat wave/{}_temp.gif".format(selected_city)
            file_ = open(gif1, "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" width="100%" alt="temp gif">',
                unsafe_allow_html=True,
            )

        with c2:
            """### Humidity trend over the decade"""
            gif2 = "images/Heat wave/{}_hum.gif".format(selected_city)
            file_1 = open(gif2, "rb")
            contents1 = file_1.read()
            data_url1 = base64.b64encode(contents1).decode("utf-8")
            file_1.close()
            st.markdown(
                f'<img src="data:image/gif;base64,{data_url1}" width="100%" alt="hum gif">',
                unsafe_allow_html=True,
            )

    else:
        """### Pollution trend over the past years"""
        info("Info", "City-wise AQI reading on a polar graph with the distance from the centre of the graph representing the value at that point of time, each rotation accounts for a full year ")
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

        with c3:

            gif1 = "images/AQI/{}_poln.gif".format(selected_city)
            file_ = open(gif1, "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" width="300%" alt="poln gif">',
                unsafe_allow_html=True,
            )


# -----Map--------
st.write("---")
st.header("Map")
info("Info", '''The map feature allows users to easily locate their desired city and access detailed information on important parameters related to Heat waves and AQI (Air Quality Index) respectively. 

It provides users with real-time data and visual representations of the values for each parameter, allowing for easy analysis and understanding of the current situation. 

''')

st.write("")
retrain_log_path = "retrain/{}/{}_retrain_log.csv".format(
    selected_model, selected_city)
df = pd.read_csv(retrain_log_path)


# Unix timestamp in seconds
unix_timestamp = df['last updated date'].iloc[-1]

# Convert Unix timestamp to datetime object
date_time = datetime.datetime.fromtimestamp(unix_timestamp)

# Format datetime object as a string
year_string = int(date_time.strftime("%Y"))
month_string = int(date_time.strftime("%m"))
date_string = int(date_time.strftime("%d"))

# Print the formatted date string


if selected_model == 'Heat wave':
    min_date = datetime.date(2012, 1, 1)
    max_date = datetime.date(year_string, month_string, date_string)
else:
    min_date = datetime.date(2020, 12, 2)
    max_date = datetime.date(year_string, month_string, date_string)

d = st.date_input(
    "Choose a date",
    datetime.date(2023, 1, 1),
    min_value=min_date,
    max_value=max_date)
with st.container():

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        path_ad = 'versioning/one/{}/1_Adilabad_data.csv'.format(
            selected_model)
        path_ka = 'versioning/one/{}/1_Karimnagar_data.csv'.format(
            selected_model)
        path_kh = 'versioning/one/{}/1_Khammam_data.csv'.format(selected_model)
        path_ni = 'versioning/one/{}/1_Nizamabad_data.csv'.format(
            selected_model)
        path_wa = 'versioning/one/{}/1_Warangal_data.csv'.format(
            selected_model)
        df_ad = pd.read_csv(path_ad)
        df_ka = pd.read_csv(path_ka)
        df_kh = pd.read_csv(path_kh)
        df_ni = pd.read_csv(path_ni)
        df_wa = pd.read_csv(path_wa)

        if selected_model == 'Heat wave':
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
                'latitude': [19.6625054, 18.6804717, 18.4348833, 17.2484683, 17.9774221],
                'longitude': [78.4953182, 78.0606503, 79.0981286, 80.006904, 79.52881]
            }

            # Convert the city data to a GeoDataFrame
            geometry = [Point(xy) for xy in zip(
                cities['longitude'], cities['latitude'])]
            cities_gdf = gpd.GeoDataFrame(
                cities, geometry=geometry, crs='EPSG:4326')

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
                'latitude': [19.6625054, 18.6804717, 18.4348833, 17.2484683, 17.9774221],
                'longitude': [78.4953182, 78.0606503, 79.0981286, 80.006904, 79.52881]
            }

            # Convert the city data to a GeoDataFrame
            geometry = [Point(xy) for xy in zip(
                cities['longitude'], cities['latitude'])]
            cities_gdf = gpd.GeoDataFrame(
                cities, geometry=geometry, crs='EPSG:4326')

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

            folium_static(m, width=520, height=520)

    with middle_column:
        st.write("                 ")

    with right_column:
        if selected_model == 'Heat wave':
            image = Image.open('images/hic1.jpeg')
            st.image(image)
        else:
            image = Image.open('images/AQI_ref.jpeg')
            st.image(image)

#-----------Description------------
st.write("---")
info("Description", "It will give the details of relevant parameters in reference to respective date and selected city.")

with st.container():
    path = "versioning/one/{}/1_{}_data.csv".format(
        selected_model, selected_city)

    if selected_model == 'Heat wave':

        df = pd.read_csv(path)
        df = heatwave_prepare(df)

        left_column, middle_column1, middle_column, right_column, middle_column2 = st.columns(
            5)
        with left_column:
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Temperature(°F) : </p>", unsafe_allow_html=True)
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Humidity : </p>", unsafe_allow_html=True)
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Heat index : </p>", unsafe_allow_html=True)
        with middle_column1:
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(
                df.loc[d, 'temp']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(
                df.loc[d, 'humidity']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(
                df.loc[d, 'heat_index']), unsafe_allow_html=True)
        with middle_column:
            st.write("                 ")
        with right_column:
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Cloud cover : </p>", unsafe_allow_html=True)
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Wind speed : </p>", unsafe_allow_html=True)
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Condition : </p>", unsafe_allow_html=True)
        with middle_column2:
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(
                df.loc[d, 'cloudcover']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(
                df.loc[d, 'windspeed']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'>{}</p>".format(
                df.loc[d, 'conditions']), unsafe_allow_html=True)
    else:
        df = pd.read_csv(path)
        df = aqi_prepare(df)
        left_column, middle_column1, right_column, middle_column2 = st.columns(
            4)
        with left_column:
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Carbon monoxide</p>", unsafe_allow_html=True)
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Nitrogen dioxide</p>", unsafe_allow_html=True)
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Ozone</p>", unsafe_allow_html=True)
        with middle_column1:
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(
                df.loc[d, 'CO']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(
                df.loc[d, 'no2']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(
                df.loc[d, 'o3']), unsafe_allow_html=True)
        with right_column:
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Sulphure dioxide</p>", unsafe_allow_html=True)
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Fine particles matter</p>", unsafe_allow_html=True)
            st.write(
                "<p style='color: #00C957; font-size: 20px;'>Ammonia</p>", unsafe_allow_html=True)
        with middle_column2:
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(
                df.loc[d, 'so2']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(
                df.loc[d, 'pm2_5']), unsafe_allow_html=True)
            st.write("<p style='color: #333333; font-size: 20px;'> : {}  μg/m3</p>".format(
                df.loc[d, 'nh3']), unsafe_allow_html=True)



st.write(
    """

    """
)


# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Get In Touch With Us")
    st.write("##")
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


