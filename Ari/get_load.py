
import numpy as np
import pandas as pd
import h5py

print keras.__version__

def transfer_learning(architecture_file=None,
                      weights_file=None,
                      n_pops=1,
                      n_train=1,
                      verbose=0):

    model = keras.models.Sequential()
    model = keras.models.model_from_json(open(architecture_file).read())
    model.load_weights(weights_file)

    # Remove the last layer
    for i in range(n_pops):
        model.pop()

    # Prevent previous layers from updating
    for l in model.layers:
            l.trainable = False

    # Add an exponential layer
    #model.add(Dense(100, activation='relu'))
    #model.add(Dense(1, activation='linear', init='normal'))
    model.add(keras.layers.core.Lambda(lambda x: x))

    model.compile(loss='poisson', optimizer='rmsprop')

    if verbose:
        for i, l in enumerate(model.get_weights()):
            print(i, np.shape(l))

    return model

vgg_model_l5 = gl.transfer_learning(architecture_file='../02-preprocessed_data/vgg16_architecture.json',
                                   weights_file='../02-preprocessed_data/vgg16_weights.h5',
                                   n_pops=5)

vgg_model_l5.save('~/Projects/02-V4py/V4py/02-preprocessed_data/')