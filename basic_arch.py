from keras import Sequential
from keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense


def build_net_1(length, channels=2):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=(length, channels)))
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def build_net_2():
    pass

def build_net_3():
    pass