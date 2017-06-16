from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, Flatten, Dense, Reshape, merge
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import LSTM
from keras import backend as K

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

    advantage = Input(shape=(1,))
    train_network = Model(input=[state, advantage], output=[value, policy])

    return value_network, policy_network, train_network, advantage

def build_feature_map(input_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
                            input_shape=input_shape, dim_ordering='th'))
    model.add(ELU(alpha=1.0))
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            subsample=(2, 2), dim_ordering='th'))
    model.add(ELU(alpha=1.0))
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            subsample=(2, 2), dim_ordering='th'))
    model.add(ELU(alpha=1.0))
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            subsample=(2, 2), dim_ordering='th'))
    model.add(ELU(alpha=1.0))
    model.add(Flatten(name="feature"))
    return model

def inverse_model(output_dim=6):
    """
    s_t, s_t+1 -> a_t
    """
    def func(ft0, ft1):
        h = merge([ft0, ft1], mode='concat')
        h = Dense(256, activation='relu')(h)
        h = Dense(output_dim, activation='sigmoid')(h)
        return h
    return func

def forward_model(output_dim=288):
    """
    s_t, a_t -> s_t+1
    """
    def func(ft, at):
        h = merge([ft, at], mode='concat')
        h = Dense(256, activation='relu')(h)
        h = Dense(output_dim, activation='linear')(h)
        return h
    return func

def build_icm_model(state_shape, action_shape, lmd=1.0, beta=0.01):
    s_t0 = Input(shape=state_shape, name="state0")
    s_t1 = Input(shape=state_shape, name="state1")
    a_t = Input(shape=action_shape, name="action")
    reshape = Reshape(target_shape=(1,) + state_shape)
    fmap = build_feature_map((1,) + state_shape)
    f_t0 = fmap(reshape(s_t0))
    f_t1 = fmap(reshape(s_t1))
    act_hat = inverse_model()(f_t0, f_t1)
    f_t1_hat = forward_model()(f_t0, a_t)
    r_in = merge([f_t1, f_t1_hat], mode=lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1),
                 output_shape=(1,), name="reward_intrinsic")
    l_i = merge([a_t, act_hat], mode=lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), axis=-1),
                output_shape=(1,))
    loss0 = merge([r_in, l_i], mode=lambda x: beta * x[0] + (1.0 - beta) * x[1], output_shape=(1,))
    rwd = Input(shape=(1,))
    loss = merge([rwd, loss0], mode=lambda x: (-lmd * x[0].T + x[1]).T, output_shape=(1,))
    return Model([s_t0, s_t1, a_t, rwd], loss)

def get_reward_intrinsic(model, x):
    return K.function([model.get_layer("state0").input,
                       model.get_layer("state1").input,
                       model.get_layer("action").input],
                      [model.get_layer("reward_intrinsic").output])(x)[0]

if __name__ == "__main__":
    import numpy as np
    icm = build_icm_model((42, 42), (6,))
    icm.summary()
    print(get_reward_intrinsic(icm, [np.zeros((1, 42, 42)), np.zeros((1, 42, 42)), np.zeros((1, 6))]))
    from keras.utils.vis_utils import plot_model
    plot_model(icm, to_file='model.png', show_shapes=True)
