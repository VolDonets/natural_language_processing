from keras.models import Sequential
from keras.layers import Conv1D


model = Sequential()
model.add(Conv1D(filters=16,
                 kernel_size=3,
                 padding='same',    # or 'valid'
                 activation='relu',
                 strides=1,
                 input_shape=(100, 300)))
