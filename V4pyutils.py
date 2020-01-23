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
#import cv2

# stats
#import pycircstat as pyc
from scipy import stats
import statsmodels.api as sm

# spykes
#from spykes.neuropop import NeuroPop
#c from spykes.neurovis import NeuroVis

# machine learning
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LabelKFold
import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Lambda
from keras.models import model_from_json
from keras.utils.layer_utils import convert_all_kernels_in_model,convert_dense_weights_data_format
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from sklearn.linear_model import LinearRegression
from keras.regularizers import l1_l2

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
    art['predictors.col'] = np.array([dat['eyes'][i]['col'] for i in dat['eyes']])
    art['predictors.row'] = np.array([dat['eyes'][i]['row'] for i in dat['eyes']])
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
            'predictors.col',
            'predictors.row',
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
    art['predictors.col'] = np.array([dat['eyes'][i]['col'] for i in dat['eyes']])
    art['predictors.row'] = np.array([dat['eyes'][i]['row'] for i in dat['eyes']])
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
            'predictors.col',
            'predictors.row',
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

    # Compute a new set of predictors
    nat_df['predictors.in_sac_blink'] = nat_df['in_sac_blink'] == 1
    nat_df['predictors.out_sac_blink'] = nat_df['out_sac_blink'] == 1

    nat_df['predictors.fix_duration'] = \
        nat_df['fix_offset'] - nat_df['fix_onset']
    nat_df['predictors.next_fix_duration'] = \
        np.append(nat_df['predictors.fix_duration'][1:], -999.0)
    nat_df['predictors.prev_fix_duration'] = \
        np.append(-999.0, nat_df['predictors.fix_duration'][0:-1])
    nat_df['predictors.row_drift'] = \
        nat_df['fix_offset_row'] - nat_df['fix_onset_row']
    nat_df['predictors.col_drift'] = \
        nat_df['fix_offset_col'] - nat_df['fix_onset_col']
    nat_df['predictors.drift'] = \
        np.abs(nat_df['predictors.row_drift']) + \
        np.abs(nat_df['predictors.col_drift'])

    # Some filters to discard bad fixations
    nat_df['filters.in_screen'] = \
        np.all((nat_df['col']>=1, nat_df['col']<=1024, \
        nat_df['row']>=1, nat_df['row']<=768), axis=0)
    nat_df['filters.badfix'] = nat_df['badfix'] == 1
    R = in_screen_radius
    nat_df['filters.in_screen_radius'] = (nat_df['row'] > R) & \
                                         (nat_df['row'] < (768 - R)) & \
                                         (nat_df['col'] > R) & \
                                         (nat_df['col'] < (1024 - R))
    # Rename some columns
    nat_df.rename(columns={'trial': 'predictors.trial',
                           'fixation': 'predictors.fixation',
                           'fix_onset': 'predictors.fix_onset',
                           'fix_offset': 'predictors.fix_offset',
                           'row': 'predictors.row',
                           'col': 'predictors.col',
                           'fix_onset_row': 'predictors.fix_onset_row',
                           'fix_onset_col': 'predictors.fix_onset_col',
                           'fix_offset_row': 'predictors.fix_offset_row',
                           'fix_offset_col': 'predictors.fix_offset_col',
                           'in_sac_dur': 'predictors.in_sac_dur',
                           'in_sac_pkvel': 'predictors.in_sac_pkvel',
                           'out_sac_dur': 'predictors.out_sac_dur',
                           'out_sac_pkvel': 'predictors.out_sac_pkvel',
                           'imname': 'im.name',
                           'impath': 'im.path'},
                   inplace=True)

    # Sort columns
    cols = ['predictors.trial', 'predictors.fixation',
            'predictors.fix_onset', 'predictors.fix_offset',
            'predictors.row', 'predictors.col',
            'predictors.fix_onset_row', 'predictors.fix_onset_col',
            'predictors.fix_offset_row', 'predictors.fix_offset_col',
            'predictors.in_sac_blink',
            'predictors.in_sac_dur',
            'predictors.in_sac_pkvel',
            'predictors.out_sac_blink',
            'predictors.out_sac_dur',
            'predictors.out_sac_pkvel',
            'predictors.fix_duration',
            'predictors.prev_fix_duration',
            'predictors.next_fix_duration',
            'predictors.row_drift',
            'predictors.col_drift',
            'predictors.drift',
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
def bgr_to_rgb(I):
    """Asserts channels last"""
    assert I.shape[2]==3
    I_copy = I.copy()
    I_copy[:,:,0] = I[:,:,2]
    I_copy[:,:,2] = I[:,:,0]
    return I_copy

def get_image(stimpath, impath, imname):
    filename = stimpath+'/'+impath+'/'+imname
    I = plt.imread(filename)
    if I is not None:
        # BGR to RGB
        I = bgr_to_rgb(I)
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
    """Resizes image using linear interpolation"""
    
    I = cv2.resize(I, imsize).astype(np.float32)
    #I = I.transpose((2,0,1))
    return I

#------------------------------------------
def get_hue_image(I):
    # Convert to Luv
    # Convert to Luv
    I = I.astype(np.float32)
    I *= 1./255
    Luv = rgb2luv(I)

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
    """Agnostic to shape of image I
    gridshape  = () 2D list or tuple of # of grids"""
    R = gridshape[0]
    C = gridshape[1]
    dtype_ = I.dtype
    b_rows = np.int(np.floor(I.shape[0]/R))
    b_cols = np.int(np.floor(I.shape[1]/C))
    Block = np.zeros([R * C, b_rows, b_cols, I.shape[2]])
    count = 0
    for i in range(R):
        for j in range(C):
            I_cut = I[b_rows * i:b_rows * (i+1),
                                     b_cols * j:b_cols *(j+1), :]
            Block[count] = I_cut
            count += 1
    Block = Block.astype(dtype_) 
    return Block

#------------------------------------------
def get_hue_histogram(hue, n_bins):
    eps = np.finfo(np.float32).eps
    h = np.histogram(hue, bins=n_bins,
                     range=(-np.pi-eps, np.pi+eps))[0]
    return h/float(np.sum(h))

from skimage.color import luv2rgb, rgb2luv

def weighted_hue_histogram(I, n_bins):
    """Returns the hue histogram of image I, but with each pixels;s hue contribution weighted by its distance
    from the LUV long axis.
    
    Inputs:
    =======
    I = RGB image"""
    
    # Convert to Luv
    I = I.astype(np.float32)
    I *= 1./255
    Luv = rgb2luv(I)
    
    # get hues
    hue = np.arctan2(Luv[:,:,2], Luv[:,:,1])
    # and "saturation": dist from the LUV central axis
    saturation = np.sqrt(np.square(Luv[:,:,2]) + np.square(Luv[:,:,1]))

    eps = np.finfo(np.float32).eps
    h = np.histogram(hue, bins=n_bins, weights=saturation,
                     range=(-np.pi-eps, np.pi+eps))[0]
    h = h/float(hue.size*100)
    
    return h

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


from skimage.color import luv2rgb, rgb2luv

def replace_with_gray(I, hue_range, saturation = 0):
    """Replaces a range of hues with isoluminant grey. Uses the LUV color space, taking U,V to some percentage
    of their original values (set with 'saturation').
    
    Inputs:
    =======
    I = RGB image
    hue_range = (low_lim, high_lim), tuple of hue range (in radians, [-pi, pi]) to bring to grey. 
                Currently no support for intervals that include 0
    saturdation  = [0,1] """
    
    # Convert to Luv
    I = I.astype(np.float32)
    I *= 1./255
    Luv = rgb2luv(I)
    
    # Take down hue, keeping luminance
    just_lum = Luv.copy()
    just_lum[:,:,1:] = saturation * just_lum[:,:,1:]
    
    # get hues
    hue = np.arctan2(Luv[:,:,2], Luv[:,:,1])
    in_range = ((hue > hue_range[0]) & (hue < hue_range[1]))
            #take this 2d array to 3d
    in_range_3d = np.broadcast_to(in_range[:,:,np.newaxis],Luv.shape)

    # return [L,0,0] if in range, or original pixel if not
    Luv_prime = np.where(in_range_3d,just_lum, Luv)
    
    # Bring back to RGB
    I_prime =   luv2rgb(Luv_prime)*255
    
    # scale back upwards by 256 and round to recover (mad annoying that this is the case.)
    I_prime = I_prime.astype(np.uint8)
    
    return I_prime

#-------------------------------------------
from tqdm import tqdm

def get_nat_features(df, 
                     reject_conditions={'filters.in_screen_radius': True, 'filters.badfix' : False},
                     stimpath='../V4pydata',
                     radius=200, RF_block=14,
                     n_histogram_bins=16,
                     model_list=['histogram'],
                     non_image_features_list=None):
    """
    This is the master function to extract image features from
    a data frame that has image name and image path
    """
    image_features = list()      # image features
    non_image_features = list()  # non-image features
    accepted_indices = list()

    if 'vgg' in model_list:
        # Instantiate vgg models
        vgg_model_l8 = vgg_transfer_ari(n_pops=0)
        vgg_model_l7 = vgg_transfer_ari(n_pops=1)
        vgg_model_l6 = vgg_transfer_ari(n_pops=2)
        vgg_model_l5 = vgg_transfer_ari(n_pops=3)

    # Loop through the data frame
    for fx in tqdm(df.index):

        # Check for reject conditions
        select = list()
        for k in reject_conditions.keys():
            select.append(df.loc[fx][k] == reject_conditions[k])
        select = np.all(select)

        prev_impath, prev_imname = None, None
        if select == True:
            # Open image file
            impath = df.loc[fx]['im.path']
            imname = df.loc[fx]['im.name']
            if(impath != prev_impath or imname != prev_imname):
                I = get_image(stimpath=stimpath, impath=impath, imname=imname)
            prev_impath, prev_imname = impath, imname
            
                        # Check for missing image file
            if I is None: 
                continue

          

            # Cut relevant fixation (maybe again)
            r, c = int(df.loc[fx]['predictors.row']), int(df.loc[fx]['predictors.col'])
            I_fix = I[r-radius:r+radius, c-radius:c+radius, :]


            # Grid image into blocks
            Block = grid_image(I_fix, [4, 4])

            this_image_feature = dict()
            # Extract feature for desired image

            if 'histogram' in model_list:
                hue_image = get_hue_image(Block[RF_block]) 
                this_image_feature['hue.histogram'] = \
                   list(get_hue_histogram(hue_image, n_bins=n_histogram_bins))
                this_image_feature['hue.mean'] = circmean(hue_image)
                this_image_feature = pd.Series(this_image_feature)

            if 'vgg' in model_list:
                # Prepare the image for vgg input
                I_fix_resize = cv2.resize(I_fix, (224,224)).astype('float32')
                I_fix_for_vgg = preprocess_input( np.expand_dims(I_fix_resize, axis=0) )

                # Compute feed forward pass
                this_image_feature['vgg.l8'] = np.squeeze(vgg_model_l8.predict(I_fix_for_vgg))
                this_image_feature['vgg.l7'] = np.squeeze(vgg_model_l7.predict(I_fix_for_vgg))
                this_image_feature['vgg.l6'] = np.squeeze(vgg_model_l6.predict(I_fix_for_vgg))
                this_image_feature['vgg.l5'] = np.squeeze(vgg_model_l5.predict(I_fix_for_vgg))

            # Accumulate non-image features
            this_non_image_feature = df.loc[fx][non_image_features_list]

            # Collect features in a list
            image_features.append(this_image_feature)
            non_image_features.append(this_non_image_feature)
            accepted_indices.append(fx)

    # Put everything into a data frame
    nat_features = pd.DataFrame({'image_features': image_features, 
                                 'non_image_features': non_image_features,
                                 'accepted_indices': accepted_indices})

    return nat_features

#--------------------------------
def nat_features_to_array(nat_features_df, image_feature='hue.histogram'):
    """
    Take a data frame containing features of interest
    and convert to array for model fitting
    """
    n_samples = len(nat_features_df)
    
    # Image features
    n_features = len(nat_features_df['image_features'][nat_features_df.index[0]][image_feature])
    image_features_array = np.zeros((n_samples, n_features))
    image_features_list = [nat_features_df['image_features'][k][image_feature] \
                           for k in nat_features_df.index]
    for k in range(n_samples):
        image_features_array[k, :] = image_features_list[k]
    
    # Non-image features
    n_features = np.shape(nat_features_df['non_image_features'][nat_features_df.index[0]].values)[0]
    non_image_features_array = np.zeros((n_samples, n_features))
    non_image_features_list = [nat_features_df['non_image_features'][k].values \
                           for k in nat_features_df.index]
    for k in range(n_samples):
        non_image_features_array[k, :] = non_image_features_list[k]

    # Concatenate
    return np.concatenate((image_features_array, 
                           non_image_features_array), 
                          axis=1)
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
    param = {'eta': 0.08340648569833894,
             'gamma': 0.2933105726714027,
             'max_depth':3,
             'num_parallel_tree': 2,
             'subsample': 0.4967204175437221}
    param['nthread'] = 8
    param['objective']= "count:poisson"
    param['eval_metric']=  "logloss"
    param['seed']=  2925
    param['silent']=  1
    param['missing']=  '-999.0'
    param['nthread'] = 8
    param['tree_method'] = 'hist'
  

    dtrain = xgb.DMatrix(Xr, label=Yr)
    dtest = xgb.DMatrix(Xt)

    num_round = 200
    bst = xgb.train(param, dtrain, num_round)

    Yt = bst.predict(dtest)
    return Yt

#---------------------------------------
def keras_GLM(input_dim, hidden_dim, learning_rate=0.0001,l1l2 = 0.01):
    model = Sequential()
    # Add a dense exponential layer with hidden_dim outputs
    if hidden_dim > 0:
        model.add(Dense(hidden_dim, input_shape=(input_dim,), \
                        kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dropout(0.5))

        # Add a dense exponential layer with 1 output
        model.add(Dense(1, kernel_initializer='glorot_normal', activation='softplus', \
                        activity_regularizer=l1_l2(l1l2)))
        
    else:
        # Add a dense exponential layer with 1 output
        model.add(Dense(1, input_shape=(input_dim,),\
                        kernel_initializer='glorot_normal', activation='softplus', \
                        activity_regularizer=l1_l2(l1l2)))

    optim = RMSprop(lr=learning_rate, clipnorm=0.5)

    model.compile(loss='poisson', optimizer=optim)
    return model

def keras_GLM_optimized(input_dim):
    

    
    dropout = 0.4803795762081918
    hidden_dim1 = 56
    hidden_dim2 = 8
    l2 = 2.2060897095713947e-08
    learning_rate = 0.0014375751342219417
    
    
    model = keras.models.Sequential()
    # Add a dense exponential layer with hidden_dim outputs

    model.add(keras.layers.Dense(int(hidden_dim1), input_shape=(input_dim,), \
                    kernel_initializer='glorot_normal', activation='relu',
                    activity_regularizer=keras.regularizers.l2(l2)))
    model.add(keras.layers.Dropout(dropout))
    
    model.add(keras.layers.normalization.BatchNormalization())
    
    model.add(keras.layers.Dense(int(hidden_dim2), \
                    kernel_initializer='glorot_normal', activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization())

    # Add a dense exponential layer with 1 output
    model.add(keras.layers.Dense(1, kernel_initializer='glorot_normal', activation='softplus', ))


    optim = keras.optimizers.adam(lr=learning_rate)

    model.compile(loss='poisson', optimizer=optim)
    return model

#---------------------------------------
def NN_poisson(Xr, Yr, Xt, batch_size, epochs, model = None):
    
    if model is None:
        model = keras_GLM_optimized(input_dim=Xr.shape[1])
        
    model.fit(Xr, Yr, batch_size=batch_size, epochs=epochs, verbose=False)
    Yt = model.predict(Xt)
    return Yt

from pyglmnet import GLM
def GLM_poisson(Xr, Yr, Xt):
    
    model = GLM(**{'distr':'softplus', 'alpha':0.1, 'tol':1e-8,
              'reg_lambda':np.logspace(np.log(0.05), np.log(0.0001), 10, base=np.exp(1)),
              'learning_rate':2, 'max_iter':10000, 'eta':2.0})
        
    model.fit(Xr, Yr)
    Yt = model[-1].predict(Xt)
    return Yt

def linear_regression(Xr,Yr,Xt,lambd=0):
    if lambd>0:
        lr = Lasso(lambd)
    else:
        lr = LinearRegression()
    lr.fit(Xr, Yr)
    Yt = lr.predict(Xt)
    
    #return rectified output (negative values cause Inf psuedo-R2)
    return np.maximum(Yt,0)

def fitted_keras(Xr, Yr, Xt,model=None):
    
    Yt = model.predict(Xt)
    return Yt

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as n2r
n2r.activate()

#---------------------------------------
def fit_cv(X, Y, algorithm = 'XGB_poisson',
           model_class=None, params = {}, batch_size=32, epochs=20,
           stratify_by_labels=[], model = None,lambd=0,
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
        

        if algorithm == 'XGB_poisson' or algorithm == 'GLM_poisson':
            Yt_hat = eval(algorithm)(Xr, Yr, Xt)
        elif algorithm == 'NN_poisson':
            model = model_class(**params)
            Yt_hat = NN_poisson(Xr, Yr, Xt, batch_size, epochs, model = model)
        elif algorithm == 'linear_regression':
            Yt_hat = eval(algorithm)(Xr, Yr, Xt,lambd=lambd)
        elif algorithm == 'fitted_keras':
            Yt_hat = fitted_keras(Xr, Yr, Xt,model = model)
        else:
            Yt_hat = model(Xr, Yr, Xt)

        Y_hat[idx_t] = Yt_hat.flatten()

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

    if verbose > 0:
        print("pseudo_R2_cv: %0.6f (+/- %0.6f)" % (np.mean(pseudo_R2_cv),
                                   np.std(pseudo_R2_cv) / np.sqrt(n_cv)))


    if verbose > 0:
        print 60 * '-'
    return Y_hat, pseudo_R2_cv

#---------------------------------------
def linear_regression(Xr,Yr,Xt,lambd=0):
    if lambd>0:
        lr = Lasso(lambd)
    else:
        lr = LinearRegression()
    lr.fit(Xr, Yr)
    Yt = lr.predict(Xt)
#     print(lr.alpha_)
    
    #return rectified output (negative values cause Inf psuedo-R2)
    return np.maximum(Yt,0)

#---------------------------------------
from sklearn.linear_model import LinearRegression, LassoCV, Lasso

def fit(X, Y, algorithm = 'XGB_poisson', model=None, batch_size=32, epochs=5, lambd = 0):
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
        def r_glmnet(Xr,Yr,Xt):
            yr = ro.FloatVector(Yr) # use factors
            trained_model = r['cv.glmnet'](Xr, yr, family="poisson",alpha=0.1,standardize=False,)
                   #    **{'lambda.min.ratio': 0.0001, 'upper.limits': 1e2, 'lower.limits': -1e2})
            lambda_min = np.asanyarray(trained_model.rx2('lambda.min'))[0]
            Yt = r['predict'](trained_model,s=lambda_min,newx=Xt,type="response")
            print(lambda_min)
            return np.array(list(Yt))

    elif algorithm == 'linear_regression':
#         
        if lambd>0:
            model = Lasso(lambd)
        else:
            model = LinearRegression()
        model.fit(X,Y)
#         print(model.alpha_)
    else:
        raise NotImplementedError('Model not implemented.')

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
    
    if K.backend() == 'theano':
        convert_all_kernels_in_model(model)

    if K.image_data_format() == 'channels_first':
        if None:
            maxpool = model.get_layer(name='block5_pool')
            shape = maxpool.output_shape[1:]
            dense = model.get_layer(name='fc1')
            convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                                  'are using the Theano '
                                  'image data format convention '
                                  '(`image_data_format="channels_first"`). '
                                  'For best performance, set '
                                  '`image_data_format="channels_last"` in '
                                  'your Keras config '
                                  'at ~/.keras/keras.json.')

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

def save_model(model, name, path = '.'):
    if not isinstance(path,basestring) or not isinstance(name,basestring):
        raise TypeError("Name and path need to be strings")
        
    json_name = path + '/' + name + '.json'
    h5_name = path + '/' + name + '.h5'
    
    model_json = model.to_json()
    with open(json_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5_name)
    print("Saved model to disk")
    
    
    
def load_model(model_name, path_to_model = '.'):
    """Assumes there is a a .h5 and .json file with name model_name
    found at path_to_model (no / necessary)"""
    if not isinstance(path_to_model,basestring) or not isinstance(model_name,basestring):
        raise TypeError("Inputs need to be strings")
        
    json_name = path_to_model + '/' + model_name + '.json'
    h5_name = path_to_model + '/' + model_name + '.h5'
    
    # load json and create model
    json_file = open(json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_name)
    print("Loaded model from disk")
    return loaded_model

def bootstrap3(x,y,labels):
    mask = np.random.choice(len(x), len(x), replace = True)
    assert mask.shape == y.shape
    
    return x[mask],y[mask], labels[mask]

def prep_data_and_fit_neurons(df_neurons, df_data, model='XGB_poisson', session = 'art', 
                         nat_features = None, image_feature = 'hue.histogram', name  = None,
                         verbose = 0, plot=False, which_neurons = 'all',joint = False,
                         simulated_responses = None, image_categories = None, category = None,
                             lambd = 0, art_XGB = None, resample=False, fit_on_halves = False):
    """
    Master function to fit neural data with specified model and predict to one-hot hue vectors to build a tuning curve.
    
    This handles a lot of the tricky data preparation. For each neuron, it looks up which trials were valid
    (this is different per neuron because we throw out sessions in which we're suspicious of neural drift for that neuron)
    and then prepares the data vectors, calls a cross-validating fitting procedure to get the quality of fit,
    then refits that model to the entire data vector, then finally uses that model to predict to the one-hot hues.
    
    There are options in this function for a lot of the control analysis, like supplying simulated data instead of 
    using real neural data, as well as fitting two models fit on only half of the model.
    
    
    
    Inputs:
    ======
    df_neurons = dataframe with neural data
    df_data = dataframe with all data. Must contain 'session_number' column
    nat_features = precomputed natural features. Result of get_nat.features
                        Required when session = 'nat'.
    
    Options:
    ========
    image_feature = {hue.histogram, hue.mean, vgg.l8, vgg.l7, vgg.l6, vgg.l5}
                        What feature to use for the natural images. 
                        Required when session = 'nat'.
                        
    model = what algorithm to feed to fit_cv 
    session = {'art', 'nat'} Which session type to fit to
    plot = {True, False} Whether to plot the fits
    verbose = {0,1,2} how much to print
    which_neurons = 'all', or list of indices of neurons to fit (in case you want just one, say)
    joint = whether to use non-images features too
    simulated_responses = list of neuron's responses. 
    valid_images = boolean vector of length nat_features.index
    resample = whether to fit to a random sample of trials, sampled with replacement
    fit_on_halves = when fitting the final model for the tuning curve analysis, 
    
    Outputs:
    ========
    df_fits: A dataframe with a single column listing, for each neuron, a dictionary with 4 value/key pairs:
    'hue', 'spike_counts', 'predicted_spike_counts', 'pseudo_R2'
    
    """
    assert session in ['art', 'nat']
    assert model in ['XGB_poisson', 'GLM_poisson','linear_regression','fitted_keras','art_model','art_model_mean_corrected']
    # handle situation where we're using the mean-firing-rate corrected art. tuning curves as the model on nat. scenes
    correct_mean = False
    if model == 'art_model_mean_corrected':
        correct_mean = True
        model = 'art_model'
        
    if session is 'nat':
        assert nat_features is not None
        assert image_feature in \
             ['hue.histogram', 'hue.mean', 'vgg.l8', 'vgg.l7', 'vgg.l6', 'vgg.l5']
    if model is 'fitted_keras':
        assert image_feature in ['vgg.l8', 'vgg.l7', 'vgg.l6', 'vgg.l5'] and session is 'nat'
    if model is 'art_model':
        assert image_feature == 'hue.histogram'
        assert art_XGB is not None
            
    if which_neurons is 'all':
        which_neurons = np.arange(len(df_neurons['name']))
    elif isinstance(which_neurons,int):
        which_neurons = [which_neurons]
        
    if image_categories is not None:
        assert image_categories.shape == (len(nat_features),)
        
    if name is None:
        name = session+'_'+model+'_'+image_feature
    
    df_fits = pd.DataFrame(columns=[name])
    which_session = session + '_sessions'
    
    # Compute feed forward features for plain hue image
    if session == 'nat':
        plain_hue = np.linspace(-np.pi, np.pi, 360)
        if image_feature in ['vgg.l8', 'vgg.l7', 'vgg.l6', 'vgg.l5']:
            
            # build model
            
            if image_feature == 'vgg.l8':
                vgg_model= vgg_transfer_ari(n_pops=0)
                n_nodes = 1011
                if not joint: n_nodes = 1000
            elif image_feature == 'vgg.l7':
                vgg_model= vgg_transfer_ari(n_pops=1)
                n_nodes = 4107
                if not joint: n_nodes = 4096
            elif image_feature == 'vgg.l6':
                vgg_model= vgg_transfer_ari(n_pops=2)
                n_nodes = 4107
                if not joint: n_nodes = 4096
            elif image_feature == 'vgg.l5':
                vgg_model= vgg_transfer_ari(n_pops=3)
                n_nodes = 25099
                if not joint: n_nodes = 25088
            
            Xplain = list()
            stimpath = '../V4pydata'
            for stim_id in range(360):
                imname = '/stimuli/M3/Hues/img%03d.jpg' % stim_id
                filename = stimpath + imname
                I = load_and_preprocess_ari(filename)
                Xplain.append(np.squeeze(vgg_model.predict(I)))
            Xplain = np.array(Xplain)
            n_bins = Xplain.shape[1]
        elif image_feature == 'hue.histogram':
            # Define histograms of plain hue stimuli
            n_bins = 16
            Xplain = onehothue(plain_hue, n_bins=n_bins)
        elif image_feature == 'hue.mean':
            Xplain = plain_hue.reshape((360,1))
    
    ######### Get tuning curves for all neurons ###########

    for neuron_id, neuron_name in tqdm(enumerate(df_neurons['name'])):
        
        if neuron_id not in which_neurons:
            continue

        if verbose>0:
            print 'Running neuron ' + neuron_name


        ### Get proper X and Y data ~~~~~~~~~~~~~~~

        # Extract session numbers
        sessions_of_interest = df_neurons.loc[neuron_id][which_session]

        # Grab relevant data
        if session is 'art':
            
            df_sessions_of_interest = df_data.loc[df_data['session.number'].isin(sessions_of_interest)]
            
            covariates =  ['predictors.hue', 
                           'predictors.col', 
                           'predictors.row', 
                           'predictors.hue_prev', 
                           'predictors.stim_dur', 
                           'predictors.off_to_onset_times']

            # Get covariates
            X = df_sessions_of_interest[covariates].values  
            # for plotting
            x_data = df_sessions_of_interest['predictors.hue'].values

            
        else:  # get natural feature
            if image_categories is not None:
                in_category = nat_features['accepted_indices'].loc[image_categories==category]
            else:
                in_category = nat_features['accepted_indices']
            
            df_sessions_of_interest = df_data.loc[df_data['session.number'].isin(sessions_of_interest) & \
                                             df_data.index.isin(in_category)]
            #-----------------
            # Get covariates
            #-----------------
            # Select sessions of interest
            indices_of_interest = np.array(df_sessions_of_interest.index)
            nat_features_of_interest = \
                nat_features.loc[nat_features['accepted_indices'].isin(indices_of_interest)]

            # Convert everything to array

            n_samples = len(nat_features_of_interest)
            
            if n_samples < 100: 
                if verbose>0: print('skipped: ', neuron_id)
                #we'll add the neuron to the datafrom but without any of the features
                #df_fits.loc[neuron_id] = 'neuron_skipped'
                continue

            # Image features
            try: n_features = len(nat_features_of_interest['image_features']\
                             [nat_features_of_interest.index[0]][image_feature])
            except TypeError: # bad practice: assuming all errors will be for same reason. 

                n_features = 1

                    
            image_features_array = np.zeros((n_samples, n_features))
            image_features_list = [nat_features_of_interest['image_features'][k][image_feature] \
                                   for k in nat_features_of_interest.index]
            for k in range(n_samples):
                image_features_array[k, :] = image_features_list[k]

            # Non-image features
            n_features = np.shape(nat_features_of_interest['non_image_features']\
                                  [nat_features_of_interest.index[0]].values)[0]
            non_image_features_array = np.zeros((n_samples, n_features))
            non_image_features_list = [nat_features_of_interest['non_image_features'][k].values \
                                   for k in nat_features_of_interest.index]
            for k in range(n_samples):
                non_image_features_array[k, :] = non_image_features_list[k]
                
            # for plotting
                            # remember we're plotting as a function of HUE MEAN
            x_data = np.array([nat_features_of_interest['image_features'][k]['hue.mean'] \
                      for k in nat_features_of_interest.index])

            # Concatenate
            if joint:
                X = np.concatenate((image_features_array, 
                                   non_image_features_array), 
                                  axis=1)
            else:
                X = image_features_array



        # Labels and number of folds for stratified CV
        labels = df_sessions_of_interest['im.name']  if session is 'nat' else []
        labels = np.array(labels)
        n_cv = 8

        # Get spike counts
        Y = df_sessions_of_interest[neuron_name].values
        
        # Rescale by the stimulus duration for all methods
        if session == 'nat':
            col = np.argwhere(nat_features['non_image_features']\
                                      [0].keys() == 'predictors.fix_duration')[0,0]
            Y = Y/non_image_features_array[:,col]
            
        
        if simulated_responses is not None:
            sim_spikes = simulated_responses[neuron_id]
            assert sim_spikes.shape == Y.shape
            Y = sim_spikes
            

        if resample is True:
            X, Y, labels = bootstrap3(X,Y,labels)
            
        if fit_on_halves:
            """Split the data in half, deterministically, making sure that trials from the same image are in the same fold."""
            half1, half2 = list(LabelKFold(labels, n_folds=2))[0]
            
            X1, Y1, labels1 = X[half1],Y[half1],labels[half1]
            X2, Y2, labels2 = X[half2],Y[half2],labels[half2]
        
 
        #### Fit models ~~~~~~~~~~~~~~~~~
        my_model = None
        
        if model == 'art_model':
            # predictions from a fit tuning curve to artificial images
            from scipy.stats import binned_statistic
            def art_model(Xr,Yr,Xt,neuron_id=neuron_id,nbins=n_bins):
                assert nbins == Xr.shape[1]
                #get average spike rate in each bin
                hue = art_XGB.iloc[:,0][neuron_id]['hue']
                spikes =   art_XGB.iloc[:,0][neuron_id]['spike_counts']/\
                           art_XGB.iloc[:,0][neuron_id]['duration']
                        
                        
                bin_means, bin_edges, binnumber = binned_statistic(hue,spikes,statistic='mean',bins=nbins)
                
                # normalize histogram by number of pixels
                Xt_unit = Xt / np.linalg.norm(Xt)

                # return lin. comb.
                prediction =  np.dot(Xt_unit, np.reshape(bin_means,(nbins,1)))
                if correct_mean:
                    art_mean = np.mean(art_XGB.iloc[:,0][neuron_id]['spike_counts']/\
                       art_XGB.iloc[:,0][neuron_id]['duration'])
                    nat_mean = np.mean(Y)
                    
                    prediction += nat_mean - art_mean
                return prediction
            my_model = art_model
            
            
        if model == 'GLM_poisson':
            r = ro.r
            r.library('glmnet')
      
            def r_glmnet(Xr,Yr,Xt):
                yr = ro.FloatVector(Yr) # use factors
                trained_model = r['cv.glmnet'](Xr, yr, family="poisson",alpha=0.1,standardize=False)

                if lambd == 'min':
                    lambda_min = np.asanyarray(trained_model.rx2('lambda.min'))[0]
                else: 
                    lambda_min = np.asanyarray(trained_model.rx2('lambda.1se'))[0]
#                print(lambda_min)
                Yt = r['predict'](trained_model,s=lambda_min,newx=Xt,type="response")
                return np.array(list(Yt))
            my_model = r_glmnet
            model = 'myGLM'
                
        if model is not 'fitted_keras':   
            # can't used fitted keras to predict for images because it was fit on those!! will be overfit
            Yt_hat, pseudo_R2 = fit_cv(X, Y,
                                      stratify_by_labels=labels,
                                      n_cv=n_cv, epochs=50,batch_size=128,
                                      algorithm= model,model = my_model,
                                      verbose=verbose, lambd = lambd)
            
            
        if model == 'myGLM':
            model = 'GLM_poisson'
        
        ### Get tuning curves
        if session is 'nat' and model is not 'art_model':
            # Fit the  model
            if model == 'fitted_keras':
                model_instance = keras_GLM(n_nodes, 100)
                if joint:
                    model_instance.load_weights('../02-preprocessed_data/M3/fit_models/'\
                                           +str(neuron_id)+'_'+neuron_name+'_l6_joint')
                else:
                    model_instance.load_weights('../02-preprocessed_data/M3/fit_models/'\
                                           +str(neuron_id)+'_'+neuron_name+'_l6_image')
            elif model is not 'GLM_poisson':
                if fit_on_halves:
                    model_instance1 = fit(X1, Y1, algorithm=model,epochs=50,batch_size=128, lambd = lambd)
                    model_instance2 = fit(X2, Y2, algorithm=model,epochs=50,batch_size=128, lambd = lambd)
                else:
                    model_instance = fit(X, Y, algorithm=model,epochs=50,batch_size=128, lambd = lambd)
            
            ### Predict on plain hue stimuli ###
            if joint:
                 # use or no?
                random_resample = np.random.randint(0, n_samples, Xplain.shape[0])
                # set :Xplain.shape[0] below to random_resample if yes
                Xplain_augment = np.concatenate((Xplain, 
                                      non_image_features_array[:Xplain.shape[0], :]), axis=1)
            else:
                Xplain_augment = Xplain
            
            if fit_on_halves:
                if model == 'XGB_poisson':
                    Yplain_hat1 = model_instance1.predict(xgb.DMatrix(Xplain_augment))
                    Yplain_hat2 = model_instance2.predict(xgb.DMatrix(Xplain_augment))
                elif model is 'GLM_poisson':
                    Yplain_hat1 = r_glmnet(X1, Y1, Xplain_augment)
                    Yplain_hat2 = r_glmnet(X2, Y2, Xplain_augment)
                else:
                    Yplain_hat1 = model_instance1.predict(Xplain_augment)
                    Yplain_hat2 = model_instance2.predict(Xplain_augment)
                
            else:
                if model == 'XGB_poisson':
                    Yplain_hat = model_instance.predict(xgb.DMatrix(Xplain_augment))
                elif model is 'GLM_poisson':
                    Yplain_hat = r_glmnet(X, Y, Xplain_augment)
                else:
                    Yplain_hat = model_instance.predict(Xplain_augment)

        
        # plot responses. for natural images, it's response to images vs. average hue of image
        if plot:
            y_data = Y
            xlabel = 'hue'
            plot_xy(x_data=x_data, y_data=y_data,
                       y_model=Yt_hat,
                       lowess_frac=0.5, xlabel=xlabel, model_name=model, 
                       x_jitter_level=0., y_jitter_level=0.5, title = 'max')
            plt.show()
            
            
        temp = dict()
        temp['hue'] = x_data
        temp['spike_counts'] = Y
        
        
        if session == 'art':
            temp['duration'] = df_sessions_of_interest['predictors.stim_dur'].values
            temp['time_normalized_counts'] = Y/df_sessions_of_interest['predictors.stim_dur'].values
            
        
        if model is not 'fitted_keras':
            temp['predicted_spike_counts'] = np.squeeze(Yt_hat)
            temp['pseudo_R2'] = pseudo_R2
                     
        if session is 'nat' and model is not 'art_model':
            temp['plain_hue'] = plain_hue
            
            if fit_on_halves:
                temp['plain_predicted_spike_counts_half1'] = np.squeeze(Yplain_hat1)
                temp['plain_predicted_spike_counts_half2'] = np.squeeze(Yplain_hat2)
            else:
                temp['plain_predicted_spike_counts'] = np.squeeze(Yplain_hat)
        
        df_fits.loc[neuron_id] = [temp]
        
    return df_fits

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
import matplotlib.collections as mcoll
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def colorline(
        x, y, z=None, cmap='hsv', norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc
def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments
def bootstrap2(x,y):
    mask = np.random.choice(len(x), len(x), replace = True)
    assert mask.shape == y.shape
    
    return x[mask],y[mask]

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


def bootstrap(yy):
    # sample with replacement
    means = []
    N = 1000
    for i in range(N):
        yy_samp = np.random.choice(yy,len(yy))
        means.append(np.nanmean(yy_samp))
    means = np.sort(means)
    crit05 = int(0.025*N)
    crit95 = int(0.975*N)
    return np.abs(np.mean(yy)-means[[crit05,crit95]])

def plot_xy(x_data=None, y_datas=None, 
            lowess_frac = 0.2,
            xlabel='variable',
            model_name='hue',
            x_jitter_level=0, y_jitter_level=0.5,
            semilogx=False,
            model_alpha=0.1,lw=4,
            colors=['#EF3E34', '#F5A21E', '#134B64',  '#02A68E', '#FF07CD'],
            data_ms = 10, title = None, labels = None,
            plot_points=True):
    
    if not isinstance(y_datas,list):
        y_datas = [y_datas]
    
    if labels is None:
        labels = ['_']*len(y_datas)

    # User lowess smoothing to smooth data and model
    lowess = sm.nonparametric.lowess
    smoothed_data = [lowess(np.squeeze(y_d), x_data, frac=lowess_frac) for y_d in y_datas]

    # Add jitter to both axes
    x_jitter = x_jitter_level * np.random.rand(np.size(x_data))
    y_jitter = y_jitter_level * np.random.rand(np.size(y_datas[0]))

    # Display
    if semilogx:
        for i,y_data in enumerate(y_datas):
            if plot_points:
                plt.semilogx(x_data + x_jitter,
                         y_data + y_jitter,
                         '.', alpha=model_alpha, color = colors[i],
                         ms=data_ms)

       

            plt.semilogx(smoothed_data[i][:, 0],
                         smoothed_data[i][:, 1],
                         color=colors[i], lw=4)


    else:
        for i,y_data in enumerate(y_datas):
            if plot_points:
                plt.plot(x_data + x_jitter,
                     y_data,
                     '.', color=colors[i],
                     alpha=model_alpha,label='_')


            plt.plot(smoothed_data[i][:,0],
                     smoothed_data[i][:,1],
                     color=colors[i], lw=lw,label=labels[i])

    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='y', right='off')
    plt.tick_params(axis='x', top='off')
    plt.xlabel(xlabel)
    plt.ylabel('spike counts')
    #plt.legend(['data', 'model %s' % model_name, 'smoothed data', 'smoothed model'],
    #           frameon=False)
    
    if title == 'max':
        model_max = smoothed_data[np.argmax(smoothed_model[:,1]),0]
        data_max = smoothed_data[np.argmax(smoothed_data[:,1]),0]
        title = 'Model {0:.2f} ; Data {1:.2f}'.format(model_max,data_max)
    
        plt.title(title)
        
    plt.legend()
  

    
## F Chollet's function to load VGG

import warnings

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
    
def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None, weights_path= '/home/klab/Projects/02-V4py/V4py/02-preprocessed_data/vgg16_weights_tf_dim_ordering_tf_kernels.h5',
          classes=1000):
    

    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.

        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """


    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def vgg_transfer_ari(n_pops=1,
                      verbose=0):
    """This loads F Chollet's keras model, and pops the top layers if wanted. """
    
    model = VGG16()

    # Remove the last layer
    #for i in range(n_pops):
    #    model.layers.pop()
        
    x=model.get_layer(index=22-n_pops).output
    
    model = Model(model.input, x)

        # Prevent previous layers from updating
    for l in model.layers:
            l.trainable = False
            
    model.compile(loss='poisson', optimizer='rmsprop')

    if verbose:
        for i, l in enumerate(model.get_weights()):
            print i, np.shape(l)

    return model


def load_and_preprocess_ari(img_path):
    """Returns an image of shape (None,224,224,3) with order BGR"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # Does RGB -> BRG here
    x = preprocess_input(x)
    return x

from scipy.stats import kendalltau as kt
#scipy.stats.weightedtau

def vectorize(mat):
    """Takes a square symmetric matrix mat and returns the vectorized form. Like matlab's squareform"""
    assert mat.shape[0]==mat.shape[1]
    
    vec = mat[:,0]
    for row in range(1,mat.shape[1]):
        vec = np.concatenate((vec,mat[row:,row]))
    
    return vec

def RDM_corr(datas, hues = False, method = 'correlation'):
    """Returns a symmetric matrix with the correlation between RDMs 
    corresponding to the data matrices in datas
    
    datas = list of high-dimensional data arrays
    hues = boolean, whether to include an artificial 'Hue similarity' matrix
    method = {'correlation','MDS','SE'}
                """
    assert isinstance(datas,list)
    
    RDMs = list()
    
    if method is 'correlation':
        embed = lambda x: np.corrcoef(x)
    elif method is 'SE':
        embed = SpectralEmbedding(n_components=2)
    elif method is 'MDS':
        embed = MDS(n_components=2)
    else:
        raise NotImplemented
    
    # get low-D representations
    for data in datas:
        if method is 'correlation':
            rdm_response = np.corrcoef(data)
        else:
            X_lowd = embed.fit_transform(Xplain_vgg)
            rdm_response = np.dot(X_lowd,X_lowd.transpose())
            
        RDMs.append(vectorize(rdm_response))
    
    if hues:
        dist_matrix = np.zeros((360,360))
        for a in range(360):
            for b in range(a,360):
                diff = 1-(b-a)/360.
                dist_matrix[a,b] = dist_matrix[b,a] = diff
        RDMs.append(vectorize(dist_matrix)) 
    
    ks = np.ones((len(RDMs),len(RDMs)))
    
    for a in range(len(RDMs)):
        for b in range(a,len(RDMs)):
            k = kt(RDMs[a],RDMs[b]).correlation
            ks[a,b] = ks[b,a] = k
            
    return ks


def get_VGG_color_tuning(df_neurons, df_data, features_location, image_feature, model = 'fitted_keras',
                        which_neurons = 'all', joint = True,
                        compare_to_response = False, nbins = 16, verbose = 0):
    """
    Builds a tuning curve from responses to images with greyed-out colors.
    
    
    Inputs
    ======
    df_neurons = dataframe with neural data
    df_data = dataframe with all data. Must contain 'session_number' column
    features_location = path to results of `get_nat_features_ari_hue_dropout`. 
                        assumes name "'precomputed_nat_features_hue_dropout_' + str(_)"
    image_feature = which layer of the VGG to use
    model = which model to train. Either 'fitted_keras' or an sklearn-style model object
    which_neurons = a list of ints of which neurons to use
    joint = To use just image features or other things too (fixation statistics, ...)
    compare_to_response = whether to subtract the responses from the actual neural response rather than the
                    predicted response. 
                    WARNING perhaps unjustified, needs evaluation
    nbins = number of color bins used to construct different images. 
            same as used when `get_nat_features_ari_hue_dropout` was called
        
    
    Outputs
    =======
    tuning_curve
    """
    assert image_feature in ['vgg.l8', 'vgg.l7', 'vgg.l6', 'vgg.l5']
    if not compare_to_response and model == 'fitted_keras':
        print('WARNING `fitted_keras` model was trained on images it is predicting for.')
    
    if which_neurons is 'all':
        which_neurons = np.arange(len(df_neurons['name']))
    elif isinstance(which_neurons,int):
        which_neurons = [which_neurons]
    
    # build model        
    if image_feature == 'vgg.l8':
        vgg_model= vgg_transfer_ari(n_pops=0)
        n_nodes = 1011
        if not joint: n_nodes = 1000
    elif image_feature == 'vgg.l7':
        vgg_model= vgg_transfer_ari(n_pops=1)
        n_nodes = 4107
        if not joint: n_nodes = 4096
    elif image_feature == 'vgg.l6':
        vgg_model= vgg_transfer_ari(n_pops=2)
        n_nodes = 4107
        if not joint: n_nodes = 4096
    elif image_feature == 'vgg.l5':
        vgg_model= vgg_transfer_ari(n_pops=3)
        n_nodes = 25099
        if not joint: n_nodes = 25088
            
    n_neurons = len(df_neurons['name'])
    assert n_neurons == 90
    hue_responses = np.zeros((n_neurons,nbins))
    
    #things to save when we look at unmodified images
    model_instances = [0]*n_neurons
    mean_responses_to_full_images = np.zeros((n_neurons,))
    
    for hue in range(nbins+1):
        if hue==0:
            this_image_feature = image_feature + '.full'
        else:
            this_image_feature = image_feature + '.lacking' + str(hue)

        file_to_load = features_location + str(hue)
        if verbose>0:
            print('... loading '+file_to_load)
        nat_features = pd.read_pickle(file_to_load)
            
        for neuron_id, neuron_name in tqdm(enumerate(df_neurons['name'])):

            if neuron_id not in which_neurons:
                continue


            ### Get X and Y data for responses to full images

            # Extract session numbers
            sessions_of_interest = df_neurons.loc[neuron_id]['nat_sessions']

            df_sessions_of_interest = df_data.loc[df_data['session.number'].isin(sessions_of_interest) & \
                                                 df_data.index.isin(nat_features['accepted_indices'])]
            #-----------------
            # Get covariates
            #-----------------
            # Select sessions of interest
            indices_of_interest = np.array(df_sessions_of_interest.index)

            nat_features_of_interest = \
                nat_features.loc[nat_features['accepted_indices'].isin(indices_of_interest)]

            # Convert everything to array

            n_samples = len(nat_features_of_interest)

            # Image features
            n_features = len(nat_features_of_interest['image_features']\
                             [nat_features_of_interest.index[0]][this_image_feature])

            image_features_array = np.zeros((n_samples, n_features))
            image_features_list = [nat_features_of_interest['image_features'][k][this_image_feature] \
                                   for k in nat_features_of_interest.index]
            assert len(image_features_list) == n_samples

            for k in range(n_samples):
                image_features_array[k, :] = image_features_list[k]

            # Non-image features
            n_features = np.shape(nat_features_of_interest['non_image_features']\
                                  [nat_features_of_interest.index[0]].values)[0]
            non_image_features_array = np.zeros((n_samples, n_features))
            non_image_features_list = [nat_features_of_interest['non_image_features'][k].values \
                                   for k in nat_features_of_interest.index\
                                  if this_image_feature in nat_features_of_interest['image_features'][k].keys()]
            for k in range(n_samples):
                non_image_features_array[k, :] = non_image_features_list[k]

            # Concatenate
            if joint:
                X = np.concatenate((image_features_array, 
                                   non_image_features_array), 
                                  axis=1)
            else:
                X = image_features_array



            # Labels and number of folds for stratified CV
            labels = df_sessions_of_interest['session.number']
            n_cv = np.size(np.unique(sessions_of_interest))
            labels = [] if n_cv == 1 else labels
            n_cv = 10 if n_cv == 1 else n_cv
            
            # first, the original images
            if hue == 0:
                # Get spike counts
                Y = df_sessions_of_interest[neuron_name].values

                #### Fit model and predict

                if model is not 'fitted_keras':   
                    # can't used fitted keras to predict for images because it was fit on those!! will be overfit
                    Yt_hat, pseudo_R2 = fit_cv(X, Y,
                                              stratify_by_labels=labels,
                                              n_cv=n_cv, epochs=50,batch_size=128,
                                              algorithm= model,model = my_model,
                                              verbose=verbose)

                    model_instance = fit(X, Y, algorithm=model,epochs=50,batch_size=128,)

                else:
                    model_instance = keras_GLM(n_nodes, 100)
                    if joint:
                        model_instance.load_weights('../02-preprocessed_data/M3/fit_models/'\
                                               +str(neuron_id)+'_'+neuron_name+'_l6_joint')
                    else:
                        model_instance.load_weights('../02-preprocessed_data/M3/fit_models/'\
                                               +str(neuron_id)+'_'+neuron_name+'_l6_image')

                    if not compare_to_response:
                        Yt_hat = model_instance.predict(X)
                
                if compare_to_response:
                    responses_to_full_images = Y
                else:
                    responses_to_full_images = Yt_hat

                #save for when we predict responses for other images
                model_instances[neuron_id] = model_instance
                mean_responses_to_full_images[neuron_id] = np.nanmean(responses_to_full_images)
        
            ### Now predict the responses to each of the minus-1-hue image sets
            # We'll record the mean difference of this response
            else:
              
                # get responses to minus-1-hue images
                responses_to_minus_1 = model_instances[neuron_id].predict(X)

                #difference of means
                hue_responses[neuron_id, hue-1] = mean_responses_to_full_images[neuron_id]\
                                                    - np.nanmean(responses_to_minus_1)
            
    return hue_responses
def bootstrap_along_axis1(yy):
    
#     print(yy.shape)
    # sample with replacement
    means = []
    N = 1000
    for i in range(N):
        mask = np.random.choice(range(yy.shape[1]), yy.shape[1])
        means.append(np.mean(yy[:,mask],axis=1))
#     print(np.array(means).shape)
    means = np.sort(np.array(means), axis=0)
    crit05 = means[50]
    crit95 = means[950]
    return crit05,crit95

def bootstrap_nan(yy):
    # sample with replacement
    means = []
    N = 1000
    for i in range(N):
        yy_samp = np.random.choice(yy,len(yy))
        means.append(np.nanmean(yy_samp))
    means = np.sort(means)
    crit05 = int(0.025*N)
    crit95 = int(0.975*N)
    return means[[crit05,crit95]]

def bootstrap_along_axis0(yy):
    
#     print(yy.shape)
    # sample with replacement
    means = []
    N = 1000
    for i in range(N):
        mask = np.random.choice(range(yy.shape[0]), yy.shape[0])
        means.append(np.mean(yy[mask],axis=0))
#     print(np.array(means).shape)
    means = np.sort(np.array(means), axis=0)
    crit05 = means[50]
    crit95 = means[950]
    return crit05,crit95

def return_number_in_bin(vector, bins = 20, lim = (-1,1)):
    bin_edges = np.arange(lim[0],lim[1], -(lim[0]-lim[1])/float(bins))
#     print(bin_edges)
    bin_counts = np.zeros(bin_edges.shape)
    number_in_bin = np.zeros(vector.shape)
    for i,value in enumerate(vector):
        # get bin index
        index = np.argwhere(value>bin_edges)[-1]
#         print(value,index)
        
        #place in bin
        bin_counts[index] += 1
        number_in_bin[i] = bin_counts[index]
    return number_in_bin
def CI_along_axis0_nan(yy, percentile=5):
    """This returns the 5 and 95% CI for a matrix, over the first index. Not the mean - the CI on the value.
    Percentile can be any integer in [1,50)
    """
    # each row is a new run; each column a neuron. So we want to sort over runs
   
    means = np.sort(yy, axis=0)
        
    crit05 = []
    crit95 = []
    for column in means.T:
        ok_values = column[~np.isnan(column)]
        percentile_index = int(len(ok_values)*percentile/100.)
        crit05.append(ok_values[percentile_index])
        crit95.append(ok_values[len(ok_values)-percentile_index])

    return np.array(crit05),np.array(crit95)

def get_slopes(neuron_id, diff_dataframe, nbins = 8,
              sats = [0,.1, 0.2, 0.3, 0.4, .5, .6,0.7, 0.75, 0.8, 0.85, 0.9,0.95, 1],
              ids = 'all', plot = False):
    """runs linear regression on the desaturation plot.
    regresses over all images for good error propagation.
    
    If ids is not None, it is a list of indexes over which to get the slopes."""
    
    slopes = []
    errs = []
    rsqs  =[]
    if plot:
#         plt.figure(figsize = (6,4))
        colors = []

    for hue in range(1,nbins+1):
        diffs = []
        stds=[]
        allpoints = []
        allsats = []
       
        for sat in sats[:-1]:
            if sat==0:
                a = diff_dataframe.loc[neuron_id,'vgg.l6.lacking' + str(hue)]
                
            else:
                a =diff_dataframe.loc[neuron_id,'vgg.l6.sat_'+str(sat)+'_bin_' + str(hue)]
            if not (ids is 'all'):
                a = a[ids]
            a = a[a**2<100]  
            diffs.append(a.mean())
            stds.append(a.std())
            allpoints.append(a)
            allsats.append(sat*np.ones(a.shape))   
            
        allpoints = np.hstack(allpoints)
        allsats = 100*(1-np.hstack(allsats))
        # get slope
        res = sm.OLS(allpoints, allsats).fit()
        slope = res.params[0]
        err = res.conf_int()[0]
        
        diffs.append(0)
        stds.append(0)
        
        rsqs.append(res.rsquared)
        slopes.append(slope)
        errs.append(err)
        if plot:
            n = len(a)
            sats_, diffs, stds = (1-np.array(sats))*100,np.array(diffs), np.array(stds)/np.sqrt(n)

            h = (hue)/8.
            c =np.roll(sns.color_palette(palette = 'hls', n_colors=8, desat=1),4)[hue-1]

            plt.plot(sats_, diffs,'-o', c=c)
            plt.fill_between(sats_, diffs-stds, diffs+stds,alpha = 0.4, color = c)
            colors.append(c)
        
    if plot:
        simpleaxis(plt.gca())
        plt.xlabel('Percent Desaturated', fontsize = 16)    
        plt.ylabel('Average difference of predictions', fontsize = 16)
        plt.tight_layout()

    #     plt.savefig('final_figures/saturation_dependence_{}.pdf'.format(neuron_id))
#         plt.show()
    return np.array(slopes), np.array(errs), np.array(rsqs)