import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# TENSORBOARD FOLDER CREATION
root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()

print(tf.__version__)
print(keras.__version__)


#  DATA LOADING
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print(X_train.shape)

X_valid, X_train = X_train[:5000]/255.0, X_train[5000:]/255.0
y_valid, y_train = y_train[:5000], y_train[5000:]

class_list = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


#  MODEL DEFINITION
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
for elem in model.layers:
    print(elem.name)


#  MODEL COMPILE AND TRAINING
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# TENSORBOARD
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

print("TRAINING MODEL ......")
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), verbose=2,
                    callbacks=[tensorboard_cb])


#  PLOTTING HISTORY
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # setting range for y
plt.savefig('classifier_plot.png')
print(df.head())


#  EVALUATING MODEL
print('EVALUATING MODEL ......')
model.evaluate(X_test, y_test)


#  PREDICTION
X_new = X_test[:5]
y_proba = model.predict(X_new)

print("Predictions for first 5 test data points...")
print(y_proba.round(2))

# class_label_id = model.predict_classes(X_new) or

class_label_id = np.argmax(y_proba, axis=1)
class_labels = np.array(class_list)[class_label_id]

y_new = np.array(y_test[:5])
ground_truth = np.array(class_list)[y_new]

print(f'Ground Truth : {ground_truth}')
print(f'Prediction : {class_labels}')
print(f'Is Correct : {ground_truth == class_labels}')
