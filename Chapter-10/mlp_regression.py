from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input

# DATA LOADING AND PREP
housing = fetch_california_housing()
scalar = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.25)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

X_train = scalar.fit_transform(X_train)
X_val = scalar.transform(X_val)
X_test = scalar.transform(X_test)

# MODEL DEFINITION
model = keras.models.Sequential()
model.add(Input(shape=X_train.shape[1:]))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

# MODEL TRAINING
model.compile(loss='mse',
              optimizer='sgd')

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))

# MODEL EVALUATION
mse_test = model.evaluate(X_test, y_test)
print(mse_test)

# MODEL PREDICTION
X_new = X_test[:5]
ground_truth = y_test[:5]

pred = model.predict(X_new)
print(f'Ground Truth : {ground_truth}')
print(f'Prediction : {pred}')
