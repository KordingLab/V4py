"""
List of utilities to make analysis of V4 data
easier and reproducible
"""
# Import dependencies
import numpy as np
import pandas as pd
from copy import deepcopy
import re
from datetime import datetime, date

# data io
import glob
import deepdish as dd

# image
import cv2

# stats
#import pycircstat as pyc
from scipy import stats
import statsmodels.api as sm

# spykes
from spykes.neuropop import NeuroPop
from spykes.neurovis import NeuroVis

# machine learning
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LabelKFold
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Lambda
from keras.models import model_from_json
from keras.regularizers import l1l2

# visualization
import matplotlib.pyplot as plt

#---------------------------------------
# Helpers for data cleaning and merging
#---------------------------------------
def art_file_to_df(session_number, session_name, neurons=None, window=[50, 300]):

    # Read in the filename
    dat = dd.io.load(session_name)

    # Collect features of interest into a dict
    art = dict()
    art['predictors.hue'] = np.array([dat['features'][i]['hue'] for i in dat['features']])
    art['predictors.onset_times'] = np.array([dat['events'][i]['onset'] for i in dat['events']])
    art['predictors.offset_times'] = np.array([dat['events'][i]['offset'] for i in dat['events']])

    # To DataFrame
    predictors_df = pd.DataFrame.from_dict(art, orient='columns', dtype=None)

    # Compute more features
    predictors_df['predictors.off_to_onset_times'] = \
            predictors_df['predictors.onset_times']- \
            np.roll(predictors_df['predictors.offset_times'], 1)
    predictors_df['predictors.off_to_onset_times'][0] = -999.0

    predictors_df['predictors.hue_prev'] = \
            np.roll(predictors_df['predictors.hue'], 1)
    predictors_df['predictors.hue_prev'][0] = -999.0

    predictors_df['predictors.stim_dur'] = \
            predictors_df['predictors.offset_times'] - \
            predictors_df['predictors.onset_times']

    # Sort columns manually
    cols = ['predictors.onset_times',
            'predictors.offset_times',
            'predictors.hue',
            'predictors.hue_prev',
            'predictors.stim_dur',
            'predictors.off_to_onset_times']
    predictors_df = predictors_df[cols]

    # Collect spike counts from neurons of interest into a dict
    all_spikecounts = dict()

    if neurons is None:
        neurons = dat['spikes'].keys()
    for neuron in neurons:
        spiketimes = dat['spikes'][neuron]
        if len(spiketimes) > 1:
            neuron_object = NeuroVis(spiketimes,
                                     name='spikes.'+neuron)
            spikecounts = \
                neuron_object.get_spikecounts(event='predictors.onset_times',
                                              df=predictors_df,
                                              window=window)
        else:
            n_samples = len(predictors_df)
            spikecounts = np.zeros(n_samples)

        all_spikecounts[neuron_object.name] = spikecounts

    # To DataFrame
    spikes_df = pd.DataFrame.from_dict(all_spikecounts, orient='columns')
    spikes_df = spikes_df[np.sort(all_spikecounts.keys())]

    # Store other metadata about the sessions
    session_df = pd.DataFrame(columns=['session.number', 'session.name'])
    session_df['session.number'] = [session_number for i in range(len(predictors_df))]
    session_df['session.name'] = [session_name for i in range(len(predictors_df))]

    # Merge
    df = pd.merge(predictors_df, spikes_df, left_index=True, right_index=True)
    df = pd.merge(df, session_df, left_index=True, right_index=True)
    return df

#------------------------------------------------------------------
def art_file_to_df_ori(session_number, session_name, neurons=None, window=[50, 300]):

    # Read in the filename
    dat = dd.io.load(session_name)

    # Collect features of interest into a dict
    art = dict()
    art['predictors.ori'] = np.array([dat['features'][i]['ori'] for i in dat['features']])
    art['predictors.onset_times'] = np.array([dat['events'][i]['onset'] for i in dat['events']])
    art['predictors.offset_times'] = np.array([dat['events'][i]['offset'] for i in dat['events']])

    # To DataFrame
    predictors_df = pd.DataFrame.from_dict(art, orient='columns', dtype=None)

    # Compute more features
    predictors_df['predictors.off_to_onset_times'] = \
            predictors_df['predictors.onset_times']- \
            np.roll(predictors_df['predictors.offset_times'], 1)
    predictors_df['predictors.off_to_onset_times'][0] = -999.0

    predictors_df['predictors.ori_prev'] = \
            np.roll(predictors_df['predictors.ori'], 1)
    predictors_df['predictors.ori_prev'][0] = -999.0

    predictors_df['predictors.stim_dur'] = \
            predictors_df['predictors.offset_times'] - \
            predictors_df['predictors.onset_times']

    # Sort columns manually
    cols = ['predictors.onset_times',
            'predictors.offset_times',
            'predictors.ori',
            'predictors.ori_prev',
            'predictors.stim_dur',
            'predictors.off_to_onset_times']
    predictors_df = predictors_df[cols]

    # Collect spike counts from neurons of interest into a dict
    all_spikecounts = dict()

    if neurons is None:
        neurons = dat['spikes'].keys()
    for neuron in neurons:
        spiketimes = dat['spikes'][neuron]
        if len(spiketimes) > 1:
            neuron_object = NeuroVis(spiketimes,
                                     name='spikes.'+neuron)
            spikecounts = \
                neuron_object.get_spikecounts(event='predictors.onset_times',
                                              df=predictors_df,
                                              window=window)
        else:
            n_samples = len(predictors_df)
            spikecounts = np.zeros(n_samples)

        all_spikecounts[neuron_object.name] = spikecounts

    # To DataFrame
    spikes_df = pd.DataFrame.from_dict(all_spikecounts, orient='columns')
    spikes_df = spikes_df[np.sort(all_spikecounts.keys())]

    # Store other metadata about the sessions
    session_df = pd.DataFrame(columns=['session.number', 'session.name'])
    session_df['session.number'] = [session_number for i in range(len(predictors_df))]
    session_df['session.name'] = [session_name for i in range(len(predictors_df))]

    # Merge
    df = pd.merge(predictors_df, spikes_df, left_index=True, right_index=True)
    df = pd.merge(df, session_df, left_index=True, right_index=True)
    return df

#------------------------------------------------------------------
def nat_file_to_df(session_number, session_name,
                   in_screen_radius=200,
                   neurons=None, window=[50, 300]):

    # Read in the file name
    dat = dd.io.load(session_name)
    features_nat = dat['eyes'][0].keys()

    # Collect features of interest into a dict
    features_dict = dict()
    for feat in features_nat:
        features_dict[feat] = \
            np.array([dat['eyes'][fix][feat] for fix in dat['eyes']])

    # Convert it into a dataframe
    nat_df = pd.DataFrame.from_dict(features_dict,
                                    orient='columns',
                                    dtype=None)

    # Add information about sessions
    nat_df['session.number'] = session_number
    nat_df['session.name'] = session_name

    # Some filters to discard bad fixations
    nat_df['filters.in_screen'] = True
    nat_df['filters.badfix'] = False
    R = in_screen_radius
    nat_df['filters.in_screen_radius'] = True
    # Rename some columns
    nat_df.rename(columns={'trial': 'predictors.trial',
                           'imname': 'im.name',
                           'impath': 'im.path'},
                   inplace=True)

    # Sort columns
    cols = ['predictors.trial',
            'predictors.stim_onset', 'predictors.stim_offset',
            'im.path', 'im.name',
            'filters.in_screen',
            'filters.in_screen_radius',
            'filters.badfix',
            'session.number',
            'session.name']
    nat_df = nat_df[cols]

    # Denote missing values correctly
    nat_df = nat_df.fillna(-999.0);

    # Collect spike counts from neurons of interest into a dict
    all_spikecounts = dict()

    if neurons is None:
        neurons = dat['spikes'].keys()
    for neuron in neurons:
        spiketimes = dat['spikes'][neuron]
        if len(spiketimes) > 1:
            neuron_object = NeuroVis(spiketimes,
                                     name='spikes.'+neuron)
            spikecounts = \
                neuron_object.get_spikecounts(event='predictors.fix_onset',
                                              df=nat_df,
                                              window=window)
        else:
            n_samples = len(nat_df)
            spikecounts = np.zeros(n_samples)

        all_spikecounts[neuron_object.name] = spikecounts

    # To DataFrame
    spikes_df = pd.DataFrame.from_dict(all_spikecounts, orient='columns')
    spikes_df = spikes_df[np.sort(all_spikecounts.keys())]

    # Merge
    df = pd.merge(nat_df, spikes_df, left_index=True, right_index=True)
    return df

# -----------------------------------------------------------------
def predict_across_sessions(df, neuron_name, Models=[], verbose=1, plot=True):

    # Get the list of unique sessions
    session_list = np.unique(df['session.number'].values)
    cross_pred_matrix = np.zeros([np.max(session_list)+1,
                                  np.max(session_list)+1])

    # Loop through pairs of sessions
    for session_i in range(np.min(session_list), np.max(session_list)+1):
        for session_j in range(session_i, np.max(session_list)+1):

            # Select only the pair of sessions of interest
            df_pair = df.loc[df['session.number'].isin([session_i, session_j])]

            # Extract spike counts
            Y = df_pair[neuron_name].values

            # If the neuron is missing in one of the sessions
            #if(np.all(Y.notnull())):
            if(np.any(np.isnan(Y)) > 0):

                # Assign cross-prediction to 0
                cross_pred_matrix[session_i, session_j] = 0

            else:

                # Extract session numbers for stratified cross-validation
                labels = df_pair['session.number']
                n_cv = np.size(np.unique(labels))

                # Loop through models
                for model_number, model in enumerate(Models):

                    if verbose == 1:
                        print 'running model %d of %d: %s' % \
                            (model_number + 1, len(Models), model)
                        print ''

                    # Extract predictors
                    X = df_pair[Models[model]['covariates']].values

                    if verbose == 1:
                        print "(", session_i, session_j, ")", \
                            "X", np.shape(X), "Y", np.shape(Y)

                    # Do cross prediction
                    # (across sessions, stratified by session)
                    if session_i != session_j:
                        Yt_hat, pseudo_R2 = fit_cv(X, Y,
                                                   stratify_by_labels=labels,
                                                   n_cv=n_cv,
                                                   algorithm='XGB_poisson',
                                                   verbose=2*verbose)

                        cross_pred_matrix[session_i, session_j] = pseudo_R2[0]
                        cross_pred_matrix[session_j, session_i] = pseudo_R2[1]

                    # Do cross-validation within session
                    elif session_i == session_j:
                        Yt_hat, pseudo_R2 = fit_cv(X, Y,
                                                   stratify_by_labels=[],
                                                   n_cv=10,
                                                   algorithm='XGB_poisson',
                                                   verbose=2*verbose)
                        cross_pred_matrix[session_i, session_i] = \
                            np.mean(pseudo_R2)

                    Models[model]['Yt_hat'] = Yt_hat
                    Models[model]['pseudo_R2'] = pseudo_R2

                    # Visualize fits
                    if plot is True:
                        x_data = df['predictors.hue'].values
                        y_data = Y
                        xlabel = 'hue'
                        plot_xy(x_data=x_data, y_data=y_data,
                                y_model=Models[model]['Yt_hat'],
                                lowess_frac=0.5,
                                xlabel=xlabel, model_name=model,
                                x_jitter_level=0., y_jitter_level=0.5)
                        plt.title((X[np.argmax(Models[model]['Yt_hat'])] * \
                                   180/np.pi)[0])
                        plt.show()

    return cross_pred_matrix

#---------------------------------------
def remove_subsets(D):
    Dnew = deepcopy(D)
    # For each element in D
    for d1 in D:
        # For all the other elements
        for d2 in D:
            if d2 != d1:
                # If d1 is contained in d2, discard d1
                if ((set(d1) | set(d2)) == set(d2)):
                    Dnew.discard(d1)

                # If d2 is contained in d1, discard d2
                elif ((set(d1) | set(d2)) == set(d1)):
                    Dnew.discard(d2)
    return Dnew

#---------------------------------------
def find_all_blocks(S):
    D = list()
    N = S.shape[0]
    for m in range(N+1):
        for n in range(m+1, N+1):
            Sblock = S[m:n,m:n]
            if (Sblock.sum() == (m - n) ** 2):
                D.append(tuple(range(m,n)))

    return list(np.sort(list(remove_subsets(set(D)))))

#---------------------------------------
def days_between(d1, d2):
    date1 = date(int(d1[0:2]), int(d1[2:4]), int(d1[4:6]))
    date2 = date(int(d2[0:2]), int(d2[2:4]), int(d2[4:6]))
    return abs(date2-date1).days

#---------------------------------------
def show_sessions(df, linked_lists, plot=False):

    art_loc_prev = -1
    nat_loc_prev = -1

    art_locs = list()
    nat_locs = list()

    # Starting date
    start_idx = df.index[0]
    date0 = str(re.findall(r'\d+', \
                re.split('_', df['Natural'].loc[start_idx])[0]))[3:9]

    # Initialize figure
    if plot:
        plt.figure(figsize=(15,4))
        ax = plt.subplot(111)
        simpleaxis(ax)

    # For each art file
    art_loc_prev = -1
    for art_file_number, art_file in enumerate(df.Hue.unique()):
        art_date = str(re.findall(r'\d+', \
                       re.split('_', art_file)[0]))[3:9]
        art_sess = np.int(str(re.findall(r'\d+', \
                          re.split('_', art_file)[-1]))[3:7])
        art_loc = days_between(art_date, date0) + 0.1 * np.float(art_sess)
        if plot:
            plt.plot(art_loc, 1, 'ro', lw=3, ms=10)

        for l in range(len(linked_lists['art'])):
                if art_file_number in linked_lists['art'][l] \
                   and art_file_number-1 in linked_lists['art'][l]:
                    if plot:
                        plt.plot(np.linspace(art_loc_prev, art_loc, 2), \
                                 [1, 1], 'r-', lw=3)

        art_loc_prev = art_loc
        art_locs.append(art_loc)

    # For each nat file
    nat_loc_prev = -1
    for nat_file_number, nat_file in enumerate(df.Natural.unique()):
        nat_date = str(re.findall(r'\d+', re.split('_', nat_file)[0]))[3:9]
        nat_sess = np.int(str(re.findall(r'\d+', \
                          re.split('_', nat_file)[-1]))[3:7])
        nat_loc = days_between(nat_date, date0) + 0.1 * np.float(nat_sess)
        if plot:
            plt.plot(nat_loc, 2, 'go', lw=3, ms=10)

        for l in range(len(linked_lists['nat'])):
                if nat_file_number in linked_lists['nat'][l] \
                   and nat_file_number-1 in linked_lists['nat'][l]:
                    if plot:
                        plt.plot(np.linspace(nat_loc_prev, nat_loc, 2), \
                                 [2, 2], 'g-', lw=3)

        nat_loc_prev = nat_loc
        nat_locs.append(nat_loc)

    # Draw vertical lines between days
    if plot:
        for xax in np.arange(25):
            plt.plot(xax * np.ones(10,), np.linspace(0.5, 2.5, 10), 'k--')

    if plot:
        plt.ylim([0.5, 2.5])
        plt.xlim([0, 25])
        #plt.axis('off')
        plt.xlabel('Days')
        plt.show()

    return art_locs, nat_locs

#---------------------------------------
def get_filenames(sessions, df, colname):
    return list(df.loc[df.index[0] + sessions][colname])

#---------------------------------------
def sessions_to_table_entry(df, neuron_name, linked_lists):

    table_entry = list()
    art_locs, nat_locs = show_sessions(df, linked_lists, plot=False)
    art_locs = np.array(art_locs)
    nat_locs = np.array(nat_locs)

    for e, l in enumerate(linked_lists['art']):
        art_sessions = np.arange(l[0], l[-1]+1)
        art_filenames = get_filenames(art_sessions, df, 'Hue')
        nat_sessions= np.where(np.all((nat_locs > art_locs[l[0]], \
                               nat_locs < art_locs[l[-1]]), axis=0))[0]
        nat_filenames = get_filenames(nat_sessions, df, 'Natural')

        table_entry.append([neuron_name,
                            art_sessions,
                            art_filenames,
                            nat_sessions,
                            nat_filenames])

    return table_entry

#---------------------------------------
# Helpers for feature extraction
#---------------------------------------
def get_firing_rate(spike_times):
    if np.size(spike_times)>1:
        fr = len(spike_times)/(spike_times[-1]-spike_times[0])
    else:
        fr = 0
    return fr

#---------------------------------------
def onehothue(theta, n_bins=16):
    eps = np.finfo(np.float32).eps
    if theta.ndim == 0:
        h = np.histogram(theta,
                         bins=n_bins,
                         range=(-np.pi-eps, np.pi+eps))[0]
    elif theta.ndim == 1:
        h = list()
        for th in theta:
            h.append(np.histogram(th,
                     bins=n_bins,
                     range=(-np.pi-eps, np.pi+eps))[0])
    else:
        print 'Error: theta has to be a scalar or 1-d array'
    h = np.array(h)
    return h

#---------------------------------------
def circkurtosis(theta):
    ck = 0.5 * (stats.kurtosis(theta) + \
         stats.kurtosis(np.arctan(np.sin(theta + np.pi),
                                         np.cos(theta + np.pi))))
    return ck

#---------------------------------------
# Helpers for image manipulation
#---------------------------------------
def get_image(stimpath, impath, imname):
    filename = stimpath+'/'+impath+'/'+imname
    I = cv2.imread(filename)
    if I is not None:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    return I

#------------------------------------------
def pad_image(I):
    [H,W,D] = I.shape
    I_pad = 128*np.ones([3*H, 3*W, D])
    I_pad[H-1:2*H-1,W-1:2*W-1,0] = I[:,:,0]
    I_pad[H-1:2*H-1,W-1:2*W-1,1] = I[:,:,1]
    I_pad[H-1:2*H-1,W-1:2*W-1,2] = I[:,:,2]
    return I_pad

#------------------------------------------
def preprocess_image(I, imsize=(224, 224)):
    I = cv2.resize(I, imsize).astype(np.float32)
    #I = I.transpose((2,0,1))
    return I

#------------------------------------------
def get_hue_image(I):
    # Convert to Luv
    I = I.astype(np.float32)
    I *= 1./255;
    Luv = cv2.cvtColor(I, cv2.COLOR_RGB2LUV);

    # Extract hue
    hue = np.arctan2(Luv[:,:,2], Luv[:,:,1])
    return hue

#------------------------------------------
def get_color_stats(im, summarize='mean'):

    im_scaled = 1.0/255.0 * im.astype(np.float32)
    im_hsv = cv2.cvtColor(im_scaled, cv2.COLOR_BGR2HSV)
    sat = im_hsv[:,:,1]
    lum = im_hsv[:,:,2]
    if summarize == 'median':
        return np.median(sat), np.median(lum)
    elif summarize == 'mean':
        return np.mean(sat), np.mean(lum)

#------------------------------------------
def visualize_hue_image(I):
    # Convert to Luv
    I = 1.0/255.0 * I.astype(np.float32)
    Luv = cv2.cvtColor(I, cv2.COLOR_RGB2LUV);

    # Assign lum channel to a constant
    Luv[:,:,0] = 60.0*np.ones([Luv.shape[0], Luv.shape[1]])

    # Convert back to RGB just to visualize
    Iviz = cv2.cvtColor(Luv, cv2.COLOR_LUV2RGB);
    return Iviz

#------------------------------------------
def grid_image(I, gridshape):
    R = gridshape[0]
    C = gridshape[1]
    b_rows = np.int(np.floor(I.shape[0]/R))
    b_cols = np.int(np.floor(I.shape[1]/C))
    Block = np.zeros([R * C, b_rows, b_cols, I.shape[2]])
    count = 0
    for i in range(R):
        for j in range(C):
            Block[count :, :, :] = I[b_rows * i:b_rows * (i+1),
                                     b_cols * j:b_cols *(j+1), :]
            count += 1
    return Block

#------------------------------------------
def get_hue_histogram(hue, n_bins):
    eps = np.finfo(np.float32).eps
    h = np.histogram(hue, bins=n_bins,
                     range=(-np.pi-eps, np.pi+eps))[0]
    return h/float(np.sum(h))

def prepare_image_for_vgg(im):
    """
    Be careful that input image should be BGR
    """
    # 01. Cast as float 32, and resize to 224 x 224 x 3
    im_for_vgg = cv2.resize(im.astype(np.float32), (224, 224))

    # 02. Subtract mean
    im_for_vgg[:,:,0] -= 103.939
    im_for_vgg[:,:,1] -= 116.779
    im_for_vgg[:,:,2] -= 123.68

    # 03. Reshape to 1 x 3 x 224 x 224
    im_for_vgg = im_for_vgg.transpose((2, 0, 1))
    im_for_vgg = np.expand_dims(im_for_vgg, axis=0)
    return im_for_vgg
#-------------------------------------------

#---------------------------------------
# Helpers for fitting models
#---------------------------------------
def poisson_pseudo_R2(y, yhat, ynull):
    y = np.squeeze(y)
    yhat = np.squeeze(yhat)
    eps = np.spacing(1)
    L1 = np.sum(y * np.log(eps + yhat) - yhat)
    L1_v = y * np.log(eps + yhat) - yhat
    L0 = np.sum(y * np.log(eps + ynull) - ynull)
    LS = np.sum(y * np.log(eps + y) - y)
    R2 = 1 - (LS - L1) / (LS - L0)
    return R2

#---------------------------------------
def XGB_poisson(Xr, Yr, Xt):
    param = {'objective': "count:poisson",
    'eval_metric': "logloss",
    'num_parallel_tree': 2,
    'eta': 0.07,
    'gamma': 1, # default = 0
    'max_depth': 2,
    'subsample': 0.5,
    'seed': 2925,
    'silent': 1,
    'missing': '-999.0'}
    param['nthread'] = 8

    dtrain = xgb.DMatrix(Xr, label=Yr)
    dtest = xgb.DMatrix(Xt)

    num_round = 200
    bst = xgb.train(param, dtrain, num_round)

    Yt = bst.predict(dtest)
    return Yt

#---------------------------------------
def keras_GLM(input_dim, hidden_dim, learning_rate=0.0001):
    model = Sequential()
    # Add a dense exponential layer with hidden_dim outputs
    model.add(Dense(hidden_dim, input_shape=(input_dim,), init='glorot_normal', activation='relu'))
    model.add(Dropout(0.5))

    # Add a dense exponential layer with 1 output
    model.add(Dense(1, init='glorot_normal', activation='softplus', W_regularizer=l1l2(l1=0.01, l2=0.01)))
    #model.add(Lambda(lambda x: np.exp(x)))

    optim = RMSprop(lr=learning_rate, clipnorm=0.5)

    model.compile(loss='poisson', optimizer=optim)
    return model

#---------------------------------------
def GLM_poisson(Xr, Yr, Xt, model, batch_size, epochs):
    #keras_glm_model = keras_GLM(input_dim=Xr.shape[1], hidden_dim=0)
    model.fit(Xr, Yr, batch_size=batch_size, nb_epoch=epochs, verbose=False)
    Yt = model.predict(Xt)
    return Yt

#---------------------------------------
def fit_cv(X, Y, algorithm = 'XGB_poisson',
           model=None, batch_size=None, epochs=None,
           stratify_by_labels=[],
           n_cv=10,
           verbose=1):
    if verbose > 0:
        print 60 * '-'
    if np.ndim(X) == 1:
        X = np.transpose(np.atleast_2d(X))

    if len(stratify_by_labels) > 0:
        skf  = LabelKFold(np.squeeze(stratify_by_labels), n_folds=n_cv)
    else:
        skf  = KFold(n=np.size(Y), n_folds=n_cv, shuffle=True, random_state=42)

    fold_count = 1
    Y_hat = np.zeros(len(Y))
    pseudo_R2_cv = list()

    for idx_r, idx_t in skf:
        if verbose > 1:
            print '...runnning cv-fold', fold_count, 'of', n_cv

        fold_count += 1

        Xr = X[idx_r, :]
        Yr = Y[idx_r]
        Xt = X[idx_t, :]
        Yt = Y[idx_t]

        if algorithm == 'XGB_poisson':
            Yt_hat = eval(algorithm)(Xr, Yr, Xt)
        elif algorithm == 'GLM_poisson':
            Yt_hat = GLM_poisson(Xr, Yr, Xt, model, batch_size, epochs)

        Y_hat[idx_t] = Yt_hat

        pseudo_R2 = poisson_pseudo_R2(Yt, Yt_hat, np.mean(Yr))
        pseudo_R2_cv.append(pseudo_R2)

        if verbose > 1:
            print 'pseudo_R2: ', pseudo_R2

    if verbose > 0:
        print("pseudo_R2_cv: %0.6f (+/- %0.6f)" % (np.mean(pseudo_R2_cv),
                                   np.std(pseudo_R2_cv) / np.sqrt(n_cv)))


    if verbose > 0:
        print 60 * '-'
    return Y_hat, pseudo_R2_cv

#---------------------------------------
def fit(X, Y, algorithm = 'XGB_poisson', model=None, batch_size=None, epochs=None):
    if algorithm=='XGB_poisson':
        param = {'objective': "count:poisson",
        'eval_metric': "logloss",
        'num_parallel_tree': 2,
        'eta': 0.07,
        'gamma': 1, # default = 0
        'max_depth': 2,
        'subsample': 0.5,
        'seed': 2925,
        'silent': 1,
        'missing': '-999.0'}
        param['nthread'] = 12

        dtrain = xgb.DMatrix(X, label=Y)

        num_round = 200
        model = xgb.train(param, dtrain, num_round)

    elif algorithm=='GLM_poisson':
        #model = keras_GLM(input_dim=Xr.shape[1], hidden_dim=100)
        model.fit(X, Y, batch_size=batch_size, nb_epoch=epochs, verbose=False)

    return model

#---------------------------------------------------------
def transfer_learning(architecture_file=None,
                      weights_file=None,
                      n_pops=1,
                      n_train=1,
                      verbose=0):

    model = Sequential()
    model = model_from_json(open(architecture_file).read())
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
    model.add(Lambda(lambda x: x))

    model.compile(loss='poisson', optimizer='rmsprop')

    if verbose:
        for i, l in enumerate(model.get_weights()):
            print i, np.shape(l)

    return model


#---------------------------------------
# Helpers for visualization
#---------------------------------------

# -----------------------------------------------------------------
def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)

# -----------------------------------------------------------------
def plot_predicted_vs_counts(models_for_plot, Y = None, models = None,
                             title = '',
                             colors=['#F5A21E', '#134B64', '#EF3E34',
                                     '#02A68E', '#FF07CD'],
                             ylim=None, simul=False):

    plt.title(title)
    unique_counts = np.unique(Y)

    for i, model in enumerate(models_for_plot):
        if simul:
            Ycounts = np.random.poisson(models[model]['Yt_hat'])
        else:
            Ycounts = Y

        if len(models_for_plot)==1:

            plt.plot(Ycounts + 0.2*np.random.normal(size=np.size(Ycounts)),
                 models[model]['Yt_hat'],
                 'k.', alpha=0.05, markersize=20)

        meanYhat = list()
        semYhat = list()

        for npks in unique_counts:
            loc = np.where(Ycounts==npks)[0]
            meanYhat.append(np.mean(models[model]['Yt_hat'][loc]))
            semYhat.append(np.std(models[model]['Yt_hat'])/np.sqrt(len(loc)))

        plt.plot(np.unique(Y), meanYhat, '.', color=colors[i],
                 ms=15, alpha=0.9)
        plt.errorbar(np.unique(Y), meanYhat, fmt='none',
                     yerr=np.array(semYhat), alpha=0.5,
                     ecolor=colors[i])

    plt.ylabel('predicted (Y_hat)')
    if simul:
        plt.xlabel('simulated spike counts (Y)')
    else:
        plt.xlabel('spike counts (Y)')

    if ylim:
        plt.ylim(ylim)
    else:
        plt.axis('equal')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='y', right='off')
    plt.tick_params(axis='x', top='off')
    if len(models_for_plot)>1:
        plt.legend(models_for_plot, frameon=False, loc=0)
    else:
        plt.legend([''] + models_for_plot, frameon=False, loc=0)


# -----------------------------------------------------------------
def plot_model_vs_model(models_for_plot, models=None, title=''):
    max_val = np.max(models[models_for_plot[1]]['Yt_hat'])
    min_val = np.min(models[models_for_plot[1]]['Yt_hat'])
    plt.plot([min_val, max_val],[min_val, max_val], '-r', lw=0.6)
    plt.plot(models[models_for_plot[0]]['Yt_hat'],
             models[models_for_plot[1]]['Yt_hat'], 'k.', alpha=0.1, ms=10)

    plt.xlabel('model ' + models_for_plot[0])
    plt.ylabel('model ' + models_for_plot[1])

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='y', right='off')
    plt.tick_params(axis='x', top='off')
    plt.title(title)
    plt.axis('equal')

# -----------------------------------------------------------------
def plot_model_comparison(models_for_plot, models=[], color='r', title=None):

    plt.plot([-1, len(models_for_plot)], [0,0],'--k', alpha=0.4)

    mean_pR2 = list()
    sem_pR2 = list()

    for model in models_for_plot:
        PR2_art = models[model]['PR2']
        mean_pR2.append(np.mean(PR2_art))
        sem_pR2.append(np.std(PR2_art)/np.sqrt(np.size(PR2_art)))

    plt.bar(np.arange(np.size(mean_pR2)), mean_pR2, 0.8, align='center',
            ecolor='k', alpha=0.3, color=color, ec='w', yerr=np.array(sem_pR2),
            tick_label=models_for_plot)
    plt.plot(np.arange(np.size(mean_pR2)), mean_pR2, 'k.', markersize=15)

    plt.ylabel('pseudo-R2')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='y', right='off')
    plt.tick_params(axis='x', top='off')
    if title:
        plt.title(title)

# -----------------------------------------------------------------
def plot_tuning_curve(models_for_plot, hues=[], Y=[], models=[], title='',
                      colors=['#F5A21E', '#134B64', '#EF3E34',
                              '#02A68E', '#FF07CD']):

    plt.plot(hues, Y + 0.01*np.random.normal(size=np.size(hues)),
             'k.', alpha=0.1, markersize=20)

    for i, model in enumerate(models_for_plot):
        plt.plot(hues, models[model]['Yt_hat'], '.',
                 color=colors[i], alpha = 0.5)

    plt.xlabel('hue angle [rad]')
    plt.ylabel('spike counts')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='y', right='off')
    plt.tick_params(axis='x', top='off')

    plt.legend(['counts (Y)'] + models_for_plot, frameon=False)
    if title:
        plt.title(title)


def plot_psth(psth, event_name='event_onset',
            condition_names=None, figsize=(8, 4), xlim=None, ylim=None,
            colors=['#F5A21E', '#134B64', '#EF3E34', '#02A68E', '#FF07CD']):
        """
        Plot psth
        Parameters
        ----------
        psth: dict, output of get_psth method
        event_name: string, legend name for event
        condition_names: list, legend names for the conditions
        figsize:
            tuple of integers, optional, default: (8, 4) width, height
            in inches.
        xlim: list
        ylim: list
        colors: list
        """


        window = psth['window']
        binsize = psth['binsize']
        conditions = psth['conditions']

        scale = 0.1
        y_min = (1.0-scale)*np.nanmin([np.min( \
            psth['data'][psth_idx]['mean']) \
            for psth_idx in psth['data']])
        y_max = (1.0+scale)*np.nanmax([np.max( \
            psth['data'][psth_idx]['mean']) \
            for psth_idx in psth['data']])

        legend = [event_name]

        time_bins = np.arange(window[0],window[1],binsize) + binsize/2.0

        if ylim:
            plt.plot([0, 0], ylim, color='k', ls='--')
        else:
            plt.plot([0, 0], [y_min, y_max], color='k', ls='--')

        for i in psth['data']:
            if np.all(np.isnan(psth['data'][i]['mean'])):
                plt.plot(0,0,alpha=1.0, color=colors[i])
            else:
                plt.plot(time_bins, psth['data'][i]['mean'],
                color=colors[i], lw=1.5)

        for i in psth['data']:
            if len(conditions) > 0:
                if condition_names:
                    legend.append(condition_names[i])
                else:
                    legend.append('Condition %d' % (i+1))
            else:
                legend.append('all')

            if not np.all(np.isnan(psth['data'][i]['mean'])):
                plt.fill_between(time_bins, psth['data'][i]['mean'] - \
                psth['data'][i]['sem'], psth['data'][i]['mean'] + \
                psth['data'][i]['sem'], color=colors[i], alpha=0.2)

        plt.xlabel('time [ms]')
        plt.ylabel('spikes per second [spks/s]')

        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        else:
            plt.ylim([y_min, y_max])

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tick_params(axis='y', right='off')
        plt.tick_params(axis='x', top='off')

        plt.legend(legend, frameon=False)


def plot_xy(x_data=None, y_data=None, y_model=None,
            lowess_frac = 0.3,
            xlabel='variable',
            model_name='hue',
            x_jitter_level=0, y_jitter_level=0.5,
            semilogx=False,
            model_alpha=0.1,
            colors=['#F5A21E', '#EF3E34', '#134B64',  '#02A68E', '#FF07CD'],
            data_ms = 10):

    # User lowess smoothing to smooth data and model
    lowess = sm.nonparametric.lowess
    smoothed_data = lowess(y_data, x_data, frac=lowess_frac)
    smoothed_model = lowess(y_model, x_data, frac=lowess_frac)

    # Add jitter to both axes
    x_jitter = x_jitter_level * np.random.rand(np.size(x_data))
    y_jitter = y_jitter_level * np.random.rand(np.size(y_data))

    # Display
    if semilogx:
        plt.semilogx(x_data + x_jitter,
                     y_data + y_jitter,
                     'k.', alpha=0.1,
                     ms=data_ms)

        plt.semilogx(x_data + x_jitter,
                     y_model,
                     '.', color=colors[1],
                     alpha=model_alpha)

        plt.semilogx(smoothed_data[:, 0],
                     smoothed_data[:, 1],
                     color='k', lw=4)

        plt.semilogx(smoothed_model[:, 0],
                     smoothed_model[:, 1],
                     color=colors[0], lw=4)
    else:
        plt.plot(x_data + x_jitter,
                 y_data + y_jitter,
                 'k.', alpha=0.1,
                 ms=data_ms)

        plt.plot(x_data + x_jitter,
                 y_model,
                 '.', color=colors[1],
                 alpha=model_alpha)

        plt.plot(smoothed_data[:,0],
                 smoothed_data[:,1],
                 color='k', lw=4)

        plt.plot(smoothed_model[:,0],
                 smoothed_model[:,1],
                 color=colors[0], lw=4)

    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='y', right='off')
    plt.tick_params(axis='x', top='off')
    plt.xlabel(xlabel)
    plt.ylabel('spike counts')
    plt.legend(['data', 'model %s' % model_name, 'smoothed data', 'smoothed model'],
               frameon=False)
