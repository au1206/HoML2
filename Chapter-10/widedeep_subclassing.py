from tensorflow import keras


class WideAndDeep(keras.Model):
    def __init__(self, units=[30,30], activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units[0], activation=activation)
        self.hidden2 = keras.layers.Dense(units[1],activation=activation)
        self.main_output = keras.layers.Dense(1, activation=activation)
        self.aux_output = keras.layers.Dense(1, activation=activation)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output


model = WideAndDeep()
model.build([(786, 1), (786, 1)])
model.compile(loss=['mse', 'mse'], optimizer='sgd')

model.summary()
