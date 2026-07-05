import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from featuring import add_distance_feature, add_rush_hour_feature, extract_datetime_features, add_weekend_feature,add_rush_hour_feature
from  visulization import plot_all_preprocessing_visuals

#ExPLORING_DATA

print('EDA - Reading and Exploring ')
df = pd.read_csv(r"D:\Training\UberPricePredictionProject\Preprocessing\uber.csv")
r1,c1=df.shape
print("Number of rows:", r1)
print("Number of columns:", c1)

print(df.iloc[:, :5])
print('--------------------')
df.info()
print('--------------------')
print(df.duplicated().sum())
print('--------------------')
print(df['fare_amount'].describe())
print('--------------------')
print(df['passenger_count'].describe())
print('--------------------')
print(df['pickup_longitude'].describe())
print('--------------------')
print(df['pickup_latitude'].describe())
print('--------------------')
print(df['dropoff_longitude'].describe())
print('--------------------')
print(df['dropoff_latitude'].describe())
print('--------------------')
zerosrows = df[
    (df['pickup_longitude'] == 0) |
    (df['pickup_latitude'] == 0) |
    (df['dropoff_longitude'] == 0) |
    (df['dropoff_latitude'] == 0)
]
print(zerosrows)

print('--------End Exploring-----------')


#Cleaning

df = df.drop(columns=['Unnamed: 0'])

df = df[df['fare_amount'] > 0]

df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 4)]

df = df.drop(columns=['key'])

df = df[(df['pickup_longitude']>=-180) & (df['pickup_longitude']<=180)]
df = df[(df['dropoff_longitude']>=-180) & (df['dropoff_longitude']<=180)]
df = df[(df['pickup_latitude']>=-90) & (df['pickup_latitude']<=90)]
df = df[(df['dropoff_latitude']>=-90) & (df['dropoff_latitude']<=90)]

df = df[df['pickup_longitude'] != 0]
df = df[df['pickup_latitude'] != 0]
df = df[df['dropoff_longitude'] != 0]
df = df[df['dropoff_latitude'] != 0]

var1 = df.isnull().sum().sum()
if var1 == 0:
    print("There is No Null values")
else: 
    print(f'There is {var1} null values')


df['dropoff_longitude']=df['dropoff_longitude'].fillna(value=df['dropoff_longitude'].mean())
df['dropoff_latitude']=df['dropoff_latitude'].fillna(value=df['dropoff_latitude'].mean())





print('--------Final results of cleaning  -----------')


print(df.info())

print(df.isnull().sum())

print("Duplicated rows:", df.duplicated().sum())

print(df.describe())

print((df[['pickup_longitude', 'pickup_latitude',
           'dropoff_longitude', 'dropoff_latitude']] == 0).sum())




print('--------Featuring-----------')

df = extract_datetime_features(df)
df = add_distance_feature(df)
df = add_weekend_feature(df)
df = add_rush_hour_feature(df)

print((df['dist_travel_km'] > 150).sum())
df = df[(df['dist_travel_km']>=0.01) & (df['dist_travel_km']<=150)]


print(df.describe())
print(df.head())


data = np.array(df)



plot_all_preprocessing_visuals(df)