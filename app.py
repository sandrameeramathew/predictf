
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import streamlit as st

# Read the data
store_sales = pd.read_csv("train.csv")
store_sales = store_sales.drop(['store', 'item'], axis=1)
store_sales['date'] = pd.to_datetime(store_sales['date'])
store_sales['date'] = store_sales['date'].dt.to_period("M")
monthly_sales = store_sales.groupby('date').sum().reset_index()
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
monthly_sales['sales_diff'] = monthly_sales['sales'].diff() 
monthly_sales = monthly_sales.dropna() 
supervised_data = monthly_sales.drop(['date', 'sales'], axis=1)
monthly_sales.info()
# Define a function to make predictions
def make_predictions(month1, month2, month3, month4, month5, month6, month7, month8, month9, month10, month11, month12):
  # Add the user input to the data
  new_row = pd.DataFrame({'month1': [month1], 'month2': [month2], 'month3': [month3], 'month4': [month4],
                          'month5': [month5], 'month6': [month6], 'month7': [month7], 'month8': [month8],
                          'month9': [month9], 'month10': [month10], 'month11': [month11], 'month12': [month12]})
  supervised_data = pd.concat([supervised_data, new_row], axis=0).reset_index(drop=True)



  # Create the supervised data
for i in range(1, 13):
 col_name = 'month' + str(i)
 supervised_data[col_name] = supervised_data['sales_diff'].shift(i) 
supervised_data = supervised_data.dropna().reset_index(drop=True)

  # Split the data into train and test sets
train_data = supervised_data[:-12]
test_data = supervised_data[-12:]

  # Scale the data
scaler = MinMaxScaler(feature_range=(-1, 1)) 
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

  # Split the data into features and target
x_train, y_train = train_data[:, 1:], train_data[:, 0:1] 
x_test, y_test = test_data[:, 1:], test_data[:, 0:1]
y_train = y_train.ravel() 
y_test = y_test.ravel()

  # Get the dates for the test set
sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)
act_sales= monthly_sales['sales'][-13:].to_list()

  # Make predictions using a linear regression model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train) 
lr_pre = lr_model.predict(x_test)
lr_pre = lr_pre.reshape(-1, 1)
lr_pre_test_set = np.concatenate([lr_pre, x_test], axis=1) 
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

  
  
# Calculate the predicted sales
result_list = []
for index in range(0, len(lr_pre_test_set)):
 result_list.append(lr_pre_test_set[index][0] + act_sales[index])

lr_pre_series = pd.Series(result_list, name="Linear Prediction")
predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index=True)

##Inverse transform the data to get actual sales
predict_df = predict_df[:-1] # drop the last row

split_date = pd.to_datetime('2016-01-01')

scaler.fit(monthly_sales.loc[monthly_sales['date'] < split_date][['sales']])
predict_df[['Linear Prediction']] = scaler.inverse_transform(predict_df[['Linear Prediction']])
predict_df = predict_df.merge(monthly_sales[['date', 'sales']], on='date')
predict_df[['sales']] = scaler.inverse_transform(predict_df[['sales']])


#Calculate evaluation metrics
mse = mean_squared_error(predict_df['sales'], predict_df['Linear Prediction'])

mae = mean_absolute_error(predict_df['sales'], predict_df['Linear Prediction'])
r2 = r2_score(predict_df['sales'], predict_df['Linear Prediction'])

print("Evaluation Metrics:")
print(f"MSE: {mse}")

print(f"MAE: {mae}")
print(f"r2: {r2}")

#Streamlit app
st.set_page_config(page_title="Customer Sales Forecast", layout="wide")

#Create a header for the app
st.write("# Customer Sales Forecast")

#Create a plot of actual and predicted sales
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(predict_df['date'], predict_df['sales'], label="Actual Sales")
ax.plot(predict_df['date'], predict_df['Linear Prediction'], label="Predicted Sales")
ax.set_title("Customer Sales Forecast using LR Model")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()

#Display the evaluation metrics
st.write("## Evaluation Metrics")
st.write(f"MSE: {mse:.2f}")
st.write(f"MAE: {mae:.2f}")
st.write(f"R2: {r2:.2f}")

#Display the predicted sales data

st.write("## Predicted Sales Data")
st.write(predict_df[['date', 'sales', 'Linear Prediction']])


