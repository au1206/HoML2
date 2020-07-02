# KERAS FUNCTIONAL API

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow import keras

wide_input_shape = (786, 1)
deep_input_shape = (786, 1)

input_a = Input(shape=wide_input_shape, name='input_wide')
input_b = Input(shape=deep_input_shape, name='input_deep')
hidden1 = Dense(50, activation='relu')(input_b)
hidden2 = Dense(30, activation='relu')(hidden1)
concat = concatenate([input_a, hidden2])
output = Dense(1, name='main_output')(concat)
aux_output = Dense(1, name='aux_output')(hidden2)

model = keras.Model(inputs=[input_a, input_b], outputs=[output, aux_output])

# individual losses and weights can be provided
model.compile(loss=['mse', 'mse'], loss_weights=[0.9, 0.1], optimizer='sgd')
model.summary()
plot_model(model, './widedeep.png')


# input and output to be provided for each input and output branch
history = model.fit([X_train_A, X_train_B],[y_train_A, y_train_B], epochs=30,
                    validation_data=([X_valid_A, X_valod_B], [y_valid_A, y_valid_B]))

# model.evaluate will now return total loss and other losses
total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B],[y_test_A, y_test_B])

#model.predict will also return all outputs
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])
