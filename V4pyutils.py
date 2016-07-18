"""
List of utilities to make analysis of V4 data
easier and reproducible
"""
# Import dependencies
import numpy as np
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
from neuropop import NeuroPop
from neurovis import NeuroVis

import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LabelKFold

import matplotlib.pyplot as plt




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
# Helpers for image manipulation
#---------------------------------------
def get_image(fx, stimpath):
    impath = fx['impath']
    imname = fx['imname']
    filename = stimpath+'/'+impath+'/'+imname
    I = cv2.imread(filename)
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
            Block[count :, :, :] = I[b_rows * i:b_rows * (i+1), b_cols * j:b_cols *(j+1), :]
            count += 1
    return Block

#------------------------------------------
def bin_it(hue, params):
    eps = np.finfo(np.float32).eps
    h = np.histogram(hue, bins=params['n_bins'],
                     range=(-np.pi-eps, np.pi+eps))[0]
    return h/float(np.sum(h))

#---------------------------------------
# Helpers for fitting models
#---------------------------------------
def poisson_logloss(y, yhat, ynull):
    eps=np.spacing(1)

    L1 = np.sum(y*np.log(eps+yhat) - yhat)
    L1_v = y*np.log(eps+yhat) - yhat
    #print np.shape(L1_v)
    L0 = np.sum(y*np.log(eps+ynull) - ynull)
    LS = np.sum(y*np.log(eps+y) - y)
    R2 = 1-(LS-L1)/(LS-L0)

    return R2

#---------------------------------------
def XGB_poisson(Xr, Yr, Xt):
    param = {'objective': "count:poisson",
    'eval_metric': "logloss",
    'num_parallel_tree': 2,
    'eta': 0.07,
    'gamma': 1, # default = 0
    'max_depth': 1,
    #'num_class': 1,
    #'colsample_bytree':0.7,
    'subsample': 0.5,
    'seed': 2925,
    'silent': 1,
    'missing': '-999.0'}
    param['nthread'] = 12

    dtrain = xgb.DMatrix(Xr, label=Yr)
    dtest = xgb.DMatrix(Xt)

    num_round = 200
    bst = xgb.train( param, dtrain, num_round )

    Yt = bst.predict( dtest )
    return Yt

#---------------------------------------
def fit_cv(X, Y, algorithm = 'XGBoost',n_cv=10,
           verbose=1, label=[]):

    if np.ndim(X)==1:
        X = np.transpose(np.atleast_2d(X))

    #skf = StratifiedKFold(Y, n_cv, shuffle=True, random_state=1)
    if len(label)>0:
        skf  = LabelKFold(np.squeeze(label), n_folds=n_cv)
    else:
        skf  = KFold(n=np.size(Y), n_folds=n_cv, shuffle=True, random_state=42)

    #skf  = LabelKFold(X['trial'].values, n_folds=n_cv)

    i=1
    Y_hat=np.zeros(len(Y))
    pR2_cv = []

    for idx_r, idx_t in skf:
        if verbose > 1:
            print '...runnning cv-fold', i, 'of', n_cv
        i+=1
        Xr = X[idx_r, :]
        Yr = Y[idx_r]
        Xt = X[idx_t, :]
        Yt = Y[idx_t]

        Yt_hat = eval(algorithm)(Xr, Yr, Xt)
        Y_hat[idx_t] = Yt_hat

        pR2 = poisson_logloss(Yt, Yt_hat, np.mean(Yr))
        pR2_cv.append(pR2)

        if verbose > 1:
            print 'pR2: ', pR2

    if verbose > 0:
        print("pR2_cv: %0.6f (+/- %0.6f)" % (np.mean(pR2_cv), np.std(pR2_cv)/np.sqrt(n_cv)))

    return Y_hat, pR2_cv

# -----------------------------------------------------------------
def plot_predicted_vs_counts(models_for_plot, Y = None, models = None,
                             title = '',
                             colors=['#F5A21E', '#134B64', '#EF3E34', '#02A68E', '#FF07CD'], ylim=None,
                             simul=False):

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

        plt.plot(np.unique(Y), meanYhat, '.', color=colors[i],  ms=15, alpha=0.9)
        plt.errorbar(np.unique(Y), meanYhat, fmt='none', yerr=np.array(semYhat), alpha=0.5, ecolor=colors[i])

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

    #plt.axis('equal')

# -----------------------------------------------------------------
def plot_model_vs_model(models_for_plot, models=None, title=''):
    max_val = np.max(models[models_for_plot[1]]['Yt_hat'])
    min_val = np.min(models[models_for_plot[1]]['Yt_hat'])
    plt.plot([min_val,max_val],[min_val,max_val], '-r', lw=0.6)
    plt.plot(models[models_for_plot[0]]['Yt_hat'], models[models_for_plot[1]]['Yt_hat'], 'k.', alpha=0.1, ms=10)

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
def plot_model_vs_time(model_for_plot, Y=None, models=None,
                       x_variable=None,
                       title=False, lowess_frac=0.1, simul=False,
                       color='r'):

    lowess = sm.nonparametric.lowess

    if simul:
        y_counts = np.random.poisson(models[model_for_plot]['Yt_hat'])
    else:
        y_counts = Y

    w = lowess(y_counts, x_variable, frac=lowess_frac)


    plt.plot(x_variable, y_counts+0.5*np.random.rand(np.size(y_counts)), 'k.', alpha=0.1, ms=10)
    plt.plot(x_variable, models[model_for_plot]['Yt_hat'], '.', c=color, alpha=0.5, lw=0.3)
    plt.plot(w[:,0], w[:,1], 'k', lw=3)

    if title:
        plt.title(title)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='y', right='off')
    plt.tick_params(axis='x', top='off')
    plt.xlabel('time [s]')
    plt.ylabel('spk counts; Yt_hat')
    if simul:
        plt.legend(['simul counts (data)','model %s' % model_for_plot,'smoothed simul data'],
                   frameon=False, loc=0)
    else:
        plt.legend(['counts (data)','model %s' % model_for_plot,'smoothed data'], frameon=False, loc=0)


# -----------------------------------------------------------------
def plot_lowess_vs_lowess(model_for_plot, Y=None, models=None,
                       x_variable='',
                       title=False, lowess_frac=0.1,
                       color='r'):

    lowess = sm.nonparametric.lowess


    y_counts_simul = np.random.poisson(models[model_for_plot]['Yt_hat'])
    y_counts = Y

    w = lowess(y_counts, x_variable, frac=lowess_frac)
    w_simul = lowess(y_counts_simul, x_variable, frac=lowess_frac)


    plt.plot(x_variable, y_counts+0.5*np.random.rand(np.size(y_counts)), 'k.', alpha=0.1, ms=10)
    plt.plot(x_variable, models[model_for_plot]['Yt_hat'], '.', c=color, alpha=0.5, lw=0.3)
    plt.plot(w[:,0], w[:,1], 'k', lw=3)
    plt.plot(w_simul[:,0], w_simul[:,1], 'm', lw=3)

    if title:
        plt.title(title)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='y', right='off')
    plt.tick_params(axis='x', top='off')
    plt.xlabel('time [s]')
    plt.ylabel('spk counts; Yt_hat')

    plt.legend(['counts (data)','model %s' % model_for_plot,'smoothed data','smoothed simul data'], frameon=False, loc=0)

# -----------------------------------------------------------------
def plot_model_comparison(models_for_plot, models=[], color='r', title=None):

    plt.plot([-1, len(models_for_plot)],[0,0],'--k', alpha=0.4)

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
                      colors=['#F5A21E', '#134B64', '#EF3E34', '#02A68E', '#FF07CD']):

    plt.plot(hues, Y + 0.01*np.random.normal(size=np.size(hues)),
             'k.', alpha=0.1, markersize=20)

    for i, model in enumerate(models_for_plot):
        plt.plot(hues, models[model]['Yt_hat'], '.', color=colors[i], alpha = 0.5)

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
