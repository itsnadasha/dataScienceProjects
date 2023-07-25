import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from prophet.plot import  plot_plotly, plot_components_plotly


BangladeshData = pd.read_csv("1901_2019_BD_weather.csv")
print(BangladeshData.head())
print(BangladeshData.tail())

print(BangladeshData.describe())
print(BangladeshData.info())

figure_temp = px.line(BangladeshData, x="Year", y="Temperature ", title="Temperature over the years in Bangladesh")
figure_temp.show()

figure_rain = px.line(BangladeshData, x="Year", y= "Rain", title="Rain in Bangladesh")
figure_rain.show()

figure_relation = px.scatter(data_frame= BangladeshData , x= "Temperature ", y="Rain", title="Relationship between Temprature and rain in bangladesh" , size="Rain" , trendline="ols")
figure_relation.show()

BangladeshData['date'] = BangladeshData['Year'].astype(str) +" "+ BangladeshData['Month'].astype(str)
BangladeshData['date'] = pd.to_datetime(BangladeshData['date'])
print(BangladeshData.head())

plt.style.use("_classic_test_patch")
plt.figure(figsize=(30,20))
plt.title("Temperature change in Bangladesh over the years")
sns.lineplot(BangladeshData, x='Month', y="Temperature " , hue="Year")
plt.show()

forecast_data = BangladeshData.rename(columns={"date" : "ds" , "Temperature " : "y"})
print(forecast_data)

model = Prophet()
model.fit(forecast_data)
forecasts = model.make_future_dataframe(periods=10)
prediction = model.predict(forecasts)
plot_plotly(model, prediction)

