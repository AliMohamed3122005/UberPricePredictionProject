# FareCast

FareCast is an app that predicts how much your Uber ride will cost before you even book it. You pick where you're starting from and where you're going on a map, and the app tells you the price.

Live demo: https://gallows-bacterium-scarce.ngrok-free.dev/

## What it does

You open the app and see a map. You click once to set your pickup point and click again to set where you're going. Then you choose how many people are riding with you and what type of ride you want (UberX, Comfort, or UberXL). Hit "Get fare estimate" and the app gives you a price, a time estimate and tells you if it's rush hour so you know why the price might be a bit higher.

Every prediction you make gets saved so there's a History tab where you can scroll back and see all your past trips. There's also a Model Comparison tab that shows how different machine learning models performed against each other so you can see the actual numbers behind the predictions instead of just trusting a black box.

## How the price gets calculated

Behind the scenes the app takes your pickup and dropoff points and works out the real distance between them, what direction you're heading, how far you are from the city center and whether you picked a busy time of day. All of that gets fed into a machine learning model that was trained on real Uber trip data and the model gives back a base price. From there the app adjusts the price depending on the ride type you picked and adds a bit extra if it's rush hour.

## Built with

**Streamlit** runs the whole app. It's a Python framework that turns a normal script into a web app without needing to write any HTML or JavaScript which made sense here since the whole team already knew Python and the focus was on the machine learning not building a frontend from scratch.

**Folium** handles the interactive map. It's a Python wrapper around Leaflet.js, so you get a real clickable map with markers and lines just by writing Python. There's a companion package called streamlit-folium that makes folium maps talk back to Streamlit, so when you click on the map, that click actually reaches the Python code.

**Pandas and NumPy** handle all the number crunching. Every time you pick a pickup and dropoff point, the app needs to calculate things like the distance between the two points and the direction of travel, and pandas makes it easy to package that into the exact same table format the model expects.

**Scikit learn** is what the model itself is built on. The joblib file that gets loaded into the app is a scikit learn model object that was trained separately, and scikit learn also provides the StandardScaler used to normalize the input features before they go into the model.

**Joblib** is used specifically to load the trained model file. It's built for saving Python objects that contain large numpy arrays, like a trained model, more efficiently than the standard pickle module.

**SQLite** stores every prediction that gets made. It doesn't need a separate database server running, the whole database is just one file, which made it the obvious choice for a project like this where you don't want to deal with setting up and hosting a real database.

## Who built it

The Streamlit application, meaning everything you interact with, the map, the interface, the database, and how it all connects to the trained model, was built by Muhamed Abdelmaboud as part of an IEEE SSCS AI team project. The rest of the team handled data cleaning and model training.

## Running it yourself

```
pip install -r requirements.txt
streamlit run app/main.py
```
