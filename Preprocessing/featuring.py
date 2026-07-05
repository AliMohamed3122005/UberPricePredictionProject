import numpy as np
from math import *
import pandas as pd


#extracting specific time
def extract_datetime_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek

    df['dayofweek']=df['dayofweek']+1

    return df


#calculate distance by longitude and latitude 

def distance(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude):
    R=6371 

    long1 = np.radians(pickup_longitude)
    lat1 = np.radians(pickup_latitude)
    long2 = np.radians(dropoff_longitude)
    lat2 = np.radians(dropoff_latitude)

    distlong = long2 - long1
    distlati = lat2 - lat1

    a=np.sin(distlati/2)**2 +np.cos(lat1)*np.cos(lat2)*np.sin(distlong/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def add_distance_feature(df):
    df = df.copy()

    df['dist_travel_km'] = distance(
        df['pickup_longitude'],
        df['pickup_latitude'],
        df['dropoff_longitude'],
        df['dropoff_latitude']
    )

    return df


def add_weekend_feature(df):
    df = df.copy()

    df['is_weekend'] = df['dayofweek'].isin([6, 7])
    return df


def add_rush_hour_feature(df):
    df = df.copy()

    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 16) & (df['hour'] <= 19))
    return df