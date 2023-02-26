import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
page_bg_img='''
<style>
body{
background-image: url(‘frs.jpg’);
baground-size:cover;
}
</style>
'''
st.markdown( page_bg_img,unsafe_allow_html=True)


# Load the sales data
sales_df = pd.read_csv('sales_data.csv')

# Create the feature matrix and target vector
X = sales_df[['item_name', 'day']]
y = sales_df['sales']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, columns=['day', 'item_name'], prefix=['day', 'item_name'])

# Create the linear regression model
model = LinearRegression()

# Train the model on the sales data
model.fit(X, y)

# Calculate the mean squared error and R^2 score for the model
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print('Mean squared error: ', mse)
print('R^2 score:', r2)

# Get input from user
item_name =st.selectbox(
    'select item name',
    ('vada', 'samoosa', 'cream bun','pazhampori','bajji'))
day = st.selectbox(
    'select item name',
    ('Monday', 'Tuesday', 'Wednesday','Thursday','Friday'))

# Create new input
X_new = pd.DataFrame({'item_name': [item_name], 'day': [day]})
X_new = pd.get_dummies(X_new, columns=['day', 'item_name'], prefix=['day', 'item_name'])

# Add missing dummy variables
missing_cols = set(X.columns) - set(X_new.columns)
for col in missing_cols:
    X_new[col] = 0

# Ensure columns are in the same order
X_new = X_new[X.columns]

# Predict the sales for the new input
y_new = model.predict(X_new)
if st.button('Predict'):
    st.write('Predicted sales: ', y_new[0])



