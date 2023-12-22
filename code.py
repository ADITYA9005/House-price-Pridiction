import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# set random seed for reproducibility
np.random.seed(42)

# get path of dataset
data_path = os.path.join(os.getcwd(), "C:\Users\HP\OneDrive\Desktop\Minor Poject Saif and team\houseprices.csv")

# load dataset
data = pd.read_csv(data_path)

# Convert categorical features into one-hot encoded variables
data = pd.get_dummies(data, columns=['Neighborhood'])

# Convert the target variable into a numeric variable
data['SalePrice'] = pd.to_numeric(data['SalePrice'])

# Split the data into input and target variables
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# perform feature scaling on input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split dataset into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# build XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.1, objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=0)

# generate XGBoost predictions for training, validation, and testing sets
xgb_train_pred = xgb_model.predict(X_train)
xgb_val_pred = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)

# define neural network architecture
def nn_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# create KerasRegressor for neural network model
nn_regressor = KerasRegressor(build_fn=nn_model, input_shape=X_train.shape[1:], batch_size=16, epochs=100, verbose=0)

# fit neural network model on training set
nn_regressor.fit(X_train, y_train)

# generate neural network predictions for training, validation, and testing sets
nn_train_pred = nn_regressor.predict(X_train)
nn_val_pred = nn_regressor.predict(X_val)
nn_test_pred = nn_regressor.predict(X_test)

# combine XGBoost and neural network predictions using simple average
ensemble_train_pred = (xgb_train_pred + nn_train_pred) / 2
ensemble_val_pred = (xgb_val_pred + nn_val_pred) / 2
ensemble_test_pred = (xgb_test_pred + nn_test_pred) / 2


# calculate root mean squared error for XGBoost, neural network, and ensemble models on training, validation, and testing sets
print("XGBoost model RMSE:")
print("Training set:", np.sqrt(mean_squared_error(y_train, xgb_train_pred)))
print("Validation set:", np.sqrt(mean_squared_error(y_val, xgb_val_pred)))
print("Testing set:", np.sqrt(mean_squared_error(y_test, xgb_test_pred)))

print("Neural network model RMSE:")
print("Training set:", np.sqrt(mean_squared_error(y_train, nn_train_pred)))
print("Validation set:", np.sqrt(mean_squared_error(y_val, nn_val_pred)))
#print("Testing set:", np.sqrt(mean_squared_error(y_test, nn_test_pred)))

print("Ensemble model RMSE:")
print("Training set:", np.sqrt(mean_squared_error(y_train, ensemble_train_pred)))
print("Validation set:", np.sqrt(mean_squared_error(y_val, ensemble_val_pred)))
#print("Testing set:", np.sqrt(mean_squared_error(y_test, ensemble_test_pred)))