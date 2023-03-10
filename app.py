
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('sales.jpg')  

store_sales=pd.read_csv("train.csv")
store_sales= store_sales.drop(['store', 'item'], axis=1)
store_sales['date'] = pd.to_datetime(store_sales['date'])
store_sales['date'] = store_sales['date'].dt.to_period("M")
monthly_sales = store_sales.groupby('date').sum().reset_index()
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
monthly_sales['sales_diff'] = monthly_sales['sales'].diff() 
monthly_sales = monthly_sales.dropna() 
supervised_data = monthly_sales.drop(['date', 'sales'],axis=1)
for i in range(1,13):
  col_name='month' + str(i)
  supervised_data[col_name] = supervised_data['sales_diff'].shift(i) 
supervised_data =supervised_data.dropna().reset_index(drop=True)
train_data = supervised_data[:-12]
test_data = supervised_data[-12:]
scaler = MinMaxScaler (feature_range=(-1,1)) 
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
x_train, y_train= train_data[:,1:],train_data[:,0:1] 
x_test, y_test =test_data[:,1:], test_data[:,0:1]
y_train =y_train.ravel() 
y_test= y_test.ravel()
sales_dates= monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)

act_sales= monthly_sales['sales'][-13:].to_list()
lr_model= LinearRegression()
lr_model.fit(x_train, y_train) 
lr_pre= lr_model.predict(x_test)
lr_pre =lr_pre.reshape(-1, 1)
lr_pre_test_set = np.concatenate([lr_pre, x_test], axis=1) 
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)
result_list =[]
for index in range(0, len(lr_pre_test_set)):
  result_list.append(lr_pre_test_set[index][0] + act_sales[index]) 
lr_pre_series = pd.Series (result_list, name="Linear Prediction")
predict_df= predict_df.merge(lr_pre_series, left_index= True, right_index=True)
lr_mse =np.sqrt(mean_squared_error(predict_df['Linear Prediction'], monthly_sales['sales'][-12:]))

lr_mae= mean_absolute_error(predict_df['Linear Prediction'], monthly_sales['sales'][-12:])

lr_r2 =r2_score (predict_df['Linear Prediction'], monthly_sales['sales'][-12:])

import streamlit as st

def run_the_app():
    st.title("Sales Prediction App")

    st.write("Enter the number of months for which you want to predict sales:")

    num_months = st.slider("Number of months", 1, 12, 1)

    st.write(f"You have selected {num_months} months")

    # Train the model with all available data
   

    # Make predictions for the selected number of months
    pred_months = monthly_sales['date'].max().to_period('M').asfreq('M', offset='M').shift(1, freq='M')
    pred_data = pd.DataFrame(columns=['sales_diff'])
    for i in range(num_months):
        pred_data.loc[i] = pred_data['sales_diff'].iloc[i-1] + model.predict(pred_data.iloc[i-1:i])[0]

    # Add predictions to the existing data and plot the results
    pred_sales = monthly_sales[['date', 'sales']].copy()
    pred_sales = pred_sales.append(pd.DataFrame({'date': pred_months.to_timestamp(),
                                                  'sales': scaler.inverse_transform(pred_data.cumsum())[-num_months:].ravel()}))
    pred_sales['date'] = pd.to_datetime(pred_sales['date'])
    pred_sales['month'] = pred_sales['date'].dt.strftime('%b-%y')

    plt.figure(figsize=(15, 5))
    plt.plot(pred_sales['date'], pred_sales['sales'])
    plt.title(f"Sales prediction for next {num_months} months")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.show()

    st.write("Here are the predicted sales for the selected number of months:")
    st.dataframe(pred_sales.tail(num_months).set_index('month'))

if __name__ == '__main__':
    run_the_app()

    
    


