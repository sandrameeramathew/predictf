import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

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

import streamlit as st
from datetime import datetime



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

split_date = pd.to_datetime('2015-01-01')

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
st.pyplot(fig)

#Display the evaluation metrics
st.write("## Evaluation Metrics")
st.write(f"MSE: {mse:.2f}")
st.write(f"MAE: {mae:.2f}")
st.write(f"R2: {r2:.2f}")

#Display the predicted sales data

st.write("## Predicted Sales Data")
st.write(predict_df[['date', 'sales', 'Linear Prediction']])

joblib.dump(lr_model, 'model.joblib')

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the trained model
trained_model = joblib.load('model.joblib')
import streamlit as st
from datetime import datetime

# Create text input widgets for each month
month1 = st.text_input('Month 1', value='')
month2 = st.text_input('Month 2', value='')
month3 = st.text_input('Month 3', value='')
month4 = st.text_input('Month 4', value='')
month5 = st.text_input('Month 5', value='')
month6 = st.text_input('Month 6', value='')
month7 = st.text_input('Month 7', value='')
month8 = st.text_input('Month 8', value='')
month9 = st.text_input('Month 9', value='')
month10 = st.text_input('Month 10', value='')
month11 = st.text_input('Month 11', value='')
month12 = st.text_input('Month 12', value='')

# Create a button to make predictions
if st.button('Make Predictions'):
    # Convert user input to float and make predictions
    try:
        month1_val = float(month1)
        month2_val = float(month2)
        month3_val = float(month3)
        month4_val = float(month4)
        month5_val = float(month5)
        month6_val = float(month6)
        month7_val = float(month7)
        month8_val = float(month8)
        month9_val = float(month9)
        month10_val = float(month10)
        month11_val = float(month11)
        month12_val = float(month12)
        
        make_predictions(month1_val, month2_val, month3_val, month4_val, month5_val,
                          month6_val, month7_val, month8_val, month9_val, month10_val,
                          month11_val, month12_val)
        
        # Display the predicted sales data
        st.write("## Predicted Sales Data")
        st.write(predict_df)
        
    except ValueError:
        st.write("Please enter valid numbers for all months.")
