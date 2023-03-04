# Team cl_AI_mate - Heat wave and AQI Prediction
# 00aa9422-7c04-4b7c-975b-6ed887ff7d95
https://sonarcloud.io/summary/overall?id=examly-test_00aa9422-7c04-4b7c-975b-6ed887ff7d95
<img src="https://drive.google.com/uc?export=view&id=1nFaRAWibLL1V4n13ATWZDanxMUr0b0Ba" alt=" " width="1010" height="100">


<a name="readme-top"></a>
<!--
***  T-AIM Academic Grand Challenge on Climate Change.


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/HarshHrs24/Team-cl_AI_mate"></a>

  <img src="https://drive.google.com/uc?export=view&id=17sCV3lHbWYeQ9JMyX9pQctvvQz3pUrgH" alt="Logo" width="150" height="100">

    
  <h3 align="center">Team cl_AI_mate</h3>

  <p align="center">
  <h4>T-AIM Academic Grand Challenge on Climate Change</h4> 
  <h3><a href="https://harshhrs24-team-cl-ai-mate-app-md7w7w.streamlit.app/"><strong>Deployed Website Link »</strong></a></h3>
    <a href="https://colab.research.google.com/drive/1xH77_KLE3gpmTxGk9-X36Pj6lHee0iBc?usp=sharing#scrollTo=1ExcTb2-2C29"><strong>Exploratory Data Analysis (EDA) for Heat wave »</strong></a>
    <br />
    <a href="https://colab.research.google.com/drive/1WgV57xtbG05shrxy47Fw59oOmzTZ2yJv?usp=sharing"><strong>Exploratory Data Analysis (EDA) for AQI »</strong></a>
    <br />
    <br />
    <a href="https://www.craft.do/s/1eTduABsPuFIDX"><strong>Complete Solution Architecture Documentation »</strong></a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#how-to-run-the-project-in-5-simple-steps">How to run the project in 5 simple steps</a>
    </li>
     <li><a href="#screenshots">Screenshots</a></li>
    <li><a href="#our-approach">Our Approach</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#file-structure">File Structure</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#our-team">Our Team</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


The problem statement asks you to build a solution to predict two environmental factors in the Tier-2 cities of the Indian state of Telangana:

<h4>1. Heat Wave Occurrences:</h4> Heat waves are prolonged periods of excessively high temperatures, which can have severe impacts on public health and local ecosystems. The task is to develop a solution that can predict when heat waves will occur in the Tier-2 cities of Telangana, to make people aware of the future occurrence of the Heat wave.

<h4>2. Air Quality Index (AQI):</h4> AQI is a measure of the air quality in a given location. It takes into account various pollutants in the air and provides a single numerical value that represents the overall air quality. The goal is to predict the AQI in the Tier-2 cities of Telangana to help residents and local authorities make informed decisions about air quality and health.

The solution should be able to accurately predict both heat wave occurrences and AQI for the time frame January 2023 - December 2023 on a monthly basis, which can help to mitigate their impacts on public health and the environment.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## How to run the project in 5 simple steps

### 1. Clone the repository or download the zip file in your machine.

### 2. Give read and write permission to workflows in your repository from Settings-->Actions-->General.

<img src="https://drive.google.com/uc?export=view&id=1DiVD8e6t15xltalqrYWa3072FRGwEB9O" alt="Click and Reload" width="600" height="300">

### 3. Generate your own personal Github Token from your profile settings-->Developer Settings-->Generate classic token--> fill the necessary requirements.

<img src="https://drive.google.com/uc?export=view&id=18dcWvUPxwmtVG9hXbw-uTbvgyO_pYPnY" alt="Click and Reload" width="1100" height="250">

### 4. Copy paste the Github token in the GitHub workflow folder-->weekly-run.yaml.

<img src="https://drive.google.com/uc?export=view&id=1qyEO2FDDxNJEpA_hz6QWcR3aJMxXnPKG" alt="Click and Reload" width="600" height="100">

### 5. Deploy the app in Streamlit.

</br>

## NOTE -
If you want to run on a local environment like VS Code first install the necessary libraries and then write the following python command in your terminal. The site will be hosted on a local host.
```sh
-m streamlit run app.py
```
To install all the libraries in the requirement.txt use the following command.
```sh
pip install -r requirements.txt 
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Screenshots 
<img src="https://drive.google.com/uc?export=view&id=1Wd50dA61aPPAf-NPgV8uvy1FNeNPWyHo" alt="Click and Reload" width="400" height="200"><img src="https://drive.google.com/uc?export=view&id=1sy6NAVGXdjpKdj2uE9OQ1wg36R-GCH_O" alt="Click and Reload" width="400" height="200">

<img src="https://drive.google.com/uc?export=view&id=1Kc31AMO-BzO_2V54ijFriwjiu0PzEmR0" alt="Click and Reload" width="400" height="200"><img src="https://drive.google.com/uc?export=view&id=1Akpxl7G-Eudbl8pknCu2gR-cIr52xAal" alt="Click and Reload" width="400" height="200">

<img src="https://drive.google.com/uc?export=view&id=1EsX3SRT0wLf49k6W4yah3kNjNExtMvGs" alt="Click and Reload" width="400" height="200"><img src="https://drive.google.com/uc?export=view&id=1-2g9sKgI5lm6rteBaQFUF-VZfL5lima2" alt="Click and Reload" width="400" height="200">

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- Approach -->
## Our Approach

The proposed solution architecture for the project based on the heat wave and AQI prediction for tier-2 cities of Telangana on a monthly basis from Jan 2023 to Dec 2023 is designed to provide an interactive dashboard with various features. The platform is scalable and business-adaptable, making it an efficient solution for the heatwave and AQI prediction.

Our platform provides forecasting for Heat wave and Air Quality Index (AQI) along with their relevant parameters for the year 2023. It offers graph visualization and analysis, highlighting major events in the year 2023 regarding the occurrence of Heat wave and severe Air Quality conditions. Users can select a specific date in 2023 to get details about temperature and AQI. The system also provides a Polar Plot to analyze past year trends and a Map feature to locate a city and get details about all necessary parameters related to Heat wave and AQI. The system follows a CI/CD architecture to constantly retrain and create new versions of the model, making it cost-effective and efficient for business scalability.

The sliding window approach used in this system involves dividing the latest month's data into four weekly windows, labeled "one" to "four". Each window contains two subfolders, "Heat wave" and "AQI," which contain the relevant raw dataset, ML model, and prediction/forecast dataset for that specific week. This approach enables the system to keep track of the latest available data and update the ML models accordingly.

To ensure that the ML models remain current, the system regularly retrains them with the latest available data. Whenever a new dataset becomes available, the raw dataset is updated, and the ML model is retrained. The content of the folder labeled "four" is then replaced by the content of the folder labeled "three," which is in turn replaced by the content of the folder labeled "two," and so on. The most recent raw dataset, ML model, and prediction/forecast dataset are then placed in the folder labeled "one". This versioning process occurs weekly, ensuring that the models are always based on the latest available data.

The system maintains a log file for these four different versions, which contains key performance indicators (KPIs) such as RMSE. These KPIs are used to compare the different versions of the ML models, and the best-performing model and prediction/forecast dataset are stored in the "winner" folder. This folder is used for forecasting heatwave and AQI, and it ensures that the system always uses the best-performing model for forecasting.

The system also implements CI/CD using Retraining and Versioning and streamlit for deployment. This allows the system to deploy the ML models without relying on expensive cloud services, making it a cost-effective way of improving the accuracy of the predictions.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Features -->
## Features
* Forecasting of Heat wave and AQI along with its relevant parameters for the year 2023.
* Graph Visualization and Analysis.
* Timeline - Highlights the major events in the year 2023 regarding occurence of Heat wave and severe Air Quality conditions.
* The user can also choose a specific date in the year 2023 to get the details about temperature and AQI. 
* Polar Plot to analyse past year trends.
* Map feature is also there on the site where the user can locate the selected city and get the details of all the necessary parameters related to Heat wave and AQI. 
* The model follows CI/CD architecture to constatntly retrain and make new versions of our model.
* Cost-effective and efficient business scalibility.
    
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## File Structure
 <img src="https://drive.google.com/uc?export=view&id=1BvXYscxxBb7qNzeHKNAostTqjh4ym_Os" alt="Logo" width="900" height="600">
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Built With

### Technology Used                                                                  
* Python
* Streamlit
* Google Colab
* GitHub
* VS Code
* JSON
* yaml
* CSS
* Excel
* Postman API
* Sonarcloud


### Libraries Used
* streamlit
* streamlit_lottie
* requests
* Pillow
* protobuf
* watchdog
* pandas
* numpy
* prophet
* neuralprophet
* tensorflow
* matplotlib
* scikit-learn
* plotly
* folium
* geopandas
* streamlit-vis-timeline
* datetime
* smtplib
* shapely.geometry

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Our Team

* [Harsh Soni](https://github.com/HarshHrs24)
* [Shivansh Rastogi](https://github.com/Shivansh1203)
* [Gaurav Garwa](https://github.com/gaurav1832)
* [Prakhar Raj Pandey](https://github.com/prakharraj1302)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments


* [T-AIM NASSCOM](https://taim-gc.in/climate-change)
* [GitHub Repository](https://github.com/iamneo-production/00aa9422-7c04-4b7c-975b-6ed887ff7d95)
* [OpenWeather](https://openweathermap.org/)
* [Visual Crossing](https://www.visualcrossing.com/)
* [India Meteorological Department - FAQ Heat wave pdf](https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf)


<p align="right">(<a href="#readme-top">back to top</a>)</p>




