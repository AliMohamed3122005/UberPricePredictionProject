# FareCast

FareCast predicts how much your Uber ride will cost before you book it. You pick a pickup point and a dropoff point on a map, and it gives you a price.

Live demo: https://gallows-bacterium-scarce.ngrok-free.dev/

## Why I built the UI the way I did

I started with the map because that's the core interaction of the whole app. Everything else, the passenger count, the ride type, the price, all depend on two points existing first. I used folium instead of Streamlit's built in map because folium wraps Leaflet.js, which is what most real map products use under the hood, and it supports click events that a plain st.map doesn't give you. On top of that I used a package called streamlit-folium, which is what actually lets a click on the map travel back into the Python session, since folium alone just renders a static map inside an iframe.

The pickup and dropoff logic runs on Streamlit's session state. First click sets the pickup point, second click sets the dropoff point, third click starts a new trip. That state has to survive across reruns, since Streamlit reruns the entire script top to bottom on every interaction, so without session state the app would forget your pickup point the instant you clicked dropoff.

I originally had all the trip inputs, passenger count, ride type, time, in Streamlit's native sidebar. I moved everything into the main page instead, because a sidebar splits attention away from the map, and a ride hailing app is supposed to feel like one continuous flow, not a form next to a separate panel.

## How the price actually gets calculated

Sixteen features go into the model for every single prediction. Five of them are raw, pickup latitude, pickup longitude, dropoff latitude, dropoff longitude, and passenger count. The other eleven get calculated the moment you pick your two points: the real distance between them using the Haversine formula, the compass bearing from pickup to dropoff, how far each point is from the city center, the hour, day, month, year, day of week, whether it's a weekend, and whether it falls inside rush hour windows.

Those sixteen numbers get scaled with the same StandardScaler that was used during training, then handed to the trained regression model, which returns a base fare. From there the app applies a ride type multiplier, UberX is the baseline, Comfort is 1.25 times that, UberXL is 1.55 times that, and adds a 1.15 times surge on top if the trip falls in a rush hour window. None of that multiplying happens inside the model itself, it's a separate pricing layer, because mixing business rules into the model would make it impossible to tell whether a bad prediction came from the model or from a pricing decision.

## Connecting to SQL

Every prediction gets written to a SQLite database the moment it happens. I chose SQLite specifically because it needs no server, no separate installation, no connection string, the entire database is one file that gets created automatically the first time the app runs.

The table has thirteen columns, pickup and dropoff coordinates, the pickup time, passenger count, ride type, trip distance, whether it was a weekend or rush hour, which model made the prediction, the predicted fare, and a timestamp. I wrapped the actual sqlite3 calls behind a repository class with two methods, save and fetch_recent, so nothing else in the app ever writes raw SQL. That means if this ever needed to move to a real database like Postgres later, only one file would need to change.

The History tab in the app is just fetch_recent being called and rendered as a table. Nothing fancy, it proves the database is really being written to, not just declared and forgotten.

## The other libraries and why

Pandas builds the single row DataFrame that gets passed into the model on every prediction, since the trained model expects a table shaped exactly like the one it was trained on, not a raw dictionary. NumPy handles the actual trigonometry behind the distance and bearing calculations, since Haversine and bearing formulas need sine, cosine, and arctangent on real coordinates.

Scikit learn is the library the model itself is built with, and it's also where the StandardScaler comes from. Joblib is what loads the trained model file back into memory, it's built specifically for saving Python objects that hold large numpy arrays more efficiently than the standard pickle module would.

## What I'd still improve

The model file currently loaded has a scaling bug from training, so predictions from it should not be trusted as final numbers yet, that's being fixed by the teammate who owns the training pipeline. The route between pickup and dropoff is currently drawn as a straight line rather than a real routed path, since adding a routing API was lower priority than getting the prediction pipeline correct first.


## Running it yourself

```
pip install -r requirements.txt
streamlit run app/main.py
```
