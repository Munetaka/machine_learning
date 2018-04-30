import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop


input_data = np.array(
    [
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ]
)

answer_data = np.array(
    [
        [0.],
        [1.],
        [1.],
        [0.]
    ]
)

model = Sequential()
model.add(Dense(2, input_shape=(2,), bias=True, activation='sigmoid'))
model.add(Dense(1, bias=True, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=RMSprop(), metrics=['accuracy'])

BATCH_SIZE = 4
ITERATION = 3000
history = model.fit(input_data, answer_data, batch_size=BATCH_SIZE, nb_epoch=ITERATION, verbose=1)

score = model.evaluate(input_data, answer_data, verbose=1)
print('score:', score[0])
print('accuracy:', score[1])

model_file_name = 'xor.json'
model_json = model.to_json()
with open(model_file_name, 'w') as file:
    file.write(model_json)

weight_file_name = 'xor_weight.hdf5'
model.save_weights(weight_file_name)

with open(model_file_name, 'r') as file:
    model_json = file.read()
    model = model_from_json(model_json)

model.load_weights(weight_file_name)

y = model.predict(np.array([[0, 0]]))
print(y)
