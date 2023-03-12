
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import streamlit as st
import joblib


@st.cache
def load_data():
    store_sales = pd.read_csv("train.csv")
    store_sales = store_sales.drop(['store', 'item'], axis=1)
    store_sales['date'] = pd.to_datetime(store_sales['date']).dt.to_period("M")
    monthly_sales = store_sales.groupby('date').sum().reset_index()
    monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
    monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
    monthly_sales = monthly_sales.dropna()
    return monthly_sales


def create_supervised_data(data):
    supervised_data = data.drop(['sales'], axis=1)
    for i in range(1, 13):
        col_name = 'month' + str(i)
        supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
    supervised_data = supervised_data.dropna().reset_index(drop=True)
    return supervised_data


def train_test_split(data):
    train_data = data[:-12]
    test_data = data[-12:]
    return train_data, test_data


def scale_data(train_data, test_data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return scaler, train_data, test_data


def get_features_targets(train_data, test_data):
    x_train, y_train = train_data[:, 1:], train_data[:, 0]
    x_test, y_test = test_data[:, 1:], test_data[:, 0]
    return x_train, y_train, x_test, y_test


def make_predictions(lr_model, scaler, x_test, act_sales):
    lr_pre = lr_model.predict(x_test.reshape(1, -1))
    lr_pre_test_set = np.concatenate([lr_pre.reshape(-1, 1), x_test.reshape(-1, 12)], axis=1)
    lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)
    result_list = [lr_pre_test_set[0][0] + act_sales[i] for i in range(len(act_sales))]
    return pd.Series(result_list, name="Linear Prediction")


def evaluate_model(predict_df):
    mse = mean_squared_error(predict_df['sales'], predict_df['Linear Prediction'])
    mae = mean_absolute_error(predict_df['sales'], predict_df['Linear Prediction'])
    r2 = r2_score(predict_df['sales'], predict_df['Linear Prediction'])
    st.write("## Evaluation Metrics")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"R2: {r2:.2f}")
    return mse, mae, r2


def plot_results(predict_df):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(predict_df['date'], predict_df['sales'], label="Actual Sales")
    ax.plot(predict_df['date'], predict_df['Linear Prediction'], label="Predicted Sales")
    ax.set_title("Customer Sales Forecast using LR Model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.write("# Customer Sales Forecast")
    st.write("## Predicted vs Actual Sales")
    st.pyplot(fig)

