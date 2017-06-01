from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, Flatten, Dense, Reshape, merge
from keras.layers.advanced_activations import ELU
from keras import backend as K

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
        h = Dense(output_dim, activation='softmax')(h)
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

def build_icm_model(state_shape, action_shape):
    s_t0 = Input(shape=state_shape)
    s_t1 = Input(shape=state_shape)
    a_t = Input(shape=action_shape)
    reshape = Reshape(target_shape=(1,) + state_shape)
    fmap = build_feature_map((1,) + state_shape)
    f_t0 = fmap(reshape(s_t0))
    f_t1 = fmap(reshape(s_t1))
    act_hat = inverse_model()(f_t0, f_t1)
    f_t1_hat = forward_model()(f_t0, a_t)
    loss = merge([f_t1, f_t1_hat], mode=lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1), output_shape=(1,))
    return Model([s_t0, s_t1, a_t], [loss, act_hat])

if __name__ == "__main__":
    import numpy as np
    icm = build_icm_model((42, 42), (6,))
    icm.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(icm, to_file='model.png', show_shapes=True)
