from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import LSTM

def build_network(input_shape, output_shape):
    state = Input(shape=input_shape)
    h = Convolution2D(32, 3, 3, border_mode='same',
                      subsample=(2, 2), dim_ordering='th')(state)
    h = ELU(alpha=1.0)(h)
    h = Convolution2D(32, 3, 3, border_mode='same',
                      subsample=(2, 2), dim_ordering='th')(h)
    h = ELU(alpha=1.0)(h)
    h = Convolution2D(32, 3, 3, border_mode='same',
                      subsample=(2, 2), dim_ordering='th')(h)
    h = ELU(alpha=1.0)(h)
    h = Convolution2D(32, 3, 3, border_mode='same',
                      subsample=(2, 2), dim_ordering='th')(h)
    h = ELU(alpha=1.0)(h)
    h = Flatten()(h)

    value = Dense(256, activation='relu')(h)
    value = Dense(1, activation='linear', name='value')(value)
    #policy = LSTM(output_shape, activation='sigmoid', name='policy')(h)
    policy = Dense(output_shape, activation='sigmoid', name='policy')(h)

    value_network = Model(input=state, output=value)
    policy_network = Model(input=state, output=policy)

    adventage = Input(shape=(1,))
    train_network = Model(input=[state, adventage], output=[value, policy])

    return value_network, policy_network, train_network, adventage
