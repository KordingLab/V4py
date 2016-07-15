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
# spykes
from neuropop import NeuroPop
from neurovis import NeuroVis

import xgboost as xgb
from sklearn.cross_validation import KFold

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
    'eta':0.07,
    'gamma':1, # default = 0
    'max_depth': 1,
    #'num_class': 1,
    #'colsample_bytree':0.7,
    'subsample': 0.5,
    'seed':2925,
    'silent':1}
    param['nthread'] = 12

    dtrain = xgb.DMatrix( Xr, label=Yr)
    dtest = xgb.DMatrix( Xt)

    num_round = 800
    bst = xgb.train( param, dtrain, num_round )

    Yt = bst.predict( dtest )
    return Yt

#---------------------------------------
def fit_cv(X, Y, algorithm = 'XGBoost',n_cv=10, silent=1):

    if np.ndim(X)==1:
        X = np.transpose(np.atleast_2d(X))

    #skf = StratifiedKFold(Y, n_cv, shuffle=True, random_state=1)
    skf  = KFold(n=np.size(Y), n_folds=n_cv, shuffle=False,random_state=None)
    #print np.shape(X[:,0])
    #skf  = LabelKFold(X['trial'].values, n_folds=n_cv)
    i=1

    Y_hat=np.zeros(len(Y))
    pR2_cv = []

    for train, test in skf:
        if not silent:
            print '...runnning cv-fold', i, 'of', n_cv
        i+=1
        idx_t = test
        idx_r = train
        #Xr = X.values[idx_r,:]
        Xr = X[idx_r,:]

        Yr = Y[idx_r]
        #Xt = X.values[idx_t,:]
        Xt = X[idx_t,:]

        Yt = Y[idx_t]

        Yt_hat = eval(algorithm)(Xr, Yr , Xt)

        Y_hat[idx_t]= Yt_hat

        pR2 = poisson_logloss(Yt, Yt_hat, np.mean(Yr))
        pR2_cv.append(pR2)

        if not silent:
            print 'pR2: ', pR2

    if not silent:
        print("pR2_cv: %0.6f (+/- %0.6f)" % (np.mean(pR2_cv), np.std(pR2_cv)/np.sqrt(n_cv)))

    return Y_hat, pR2_cv
