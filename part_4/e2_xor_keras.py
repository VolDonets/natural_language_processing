import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


# for solving XOR problem
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

model = Sequential()
num_neurons = 10
model.add(Dense(num_neurons, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

sgd = SGD(learning_rate=0.5)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
print('Model prediction before training:')
print(model.predict(x_train))

print("\n\nLet's train the model:")
model.fit(x_train, y_train, epochs=100)

# saving the model:
model_structure = model.to_json()
with open('../src/xor_keras/structure_xor_keras_model.json', 'w') as json_file:
    json_file.write(model_structure)
model.save_weights('../src/xor_keras/weights_xor_keras_model.h5')

