import numpy as np
import pandas as pd
import pathlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import glob
import random
import time
import joblib

import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasRegressor
from tensorflow.keras import Input, Model
from tensorflow.keras import layers, models, initializers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import datasets
from sklearn.metrics import make_scorer

#define rootdir as path to this script
rootdir = pathlib.Path(__file__ ).parent
#define directory where data is held
datadir = rootdir.parent/"DATA_DIRECTORY"
#define directory where results will be output
resultdir = rootdir/"RESULT_OUTPUT_DIRECTORY"

###define functions for predicting various stats to be used in model.compile(metrics=[])
#both numpy and tensorflow formats are required for this script
#these 4 functions are for numpy formats
def r2_func(y_true, y_pred, **kwargs):
    return metrics.r2_score(y_true, y_pred)
def rmse_func(y_true, y_pred, **kwargs):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))  
def bias_func(y_true, y_pred, **kwargs):
    return np.mean(y_true-y_pred)
def sdep_func(y_true, y_pred, **kwargs):
    return (np.mean((y_true-y_pred-(np.mean(y_true-y_pred)))**2))**0.5
#these 4 are for tensorflow formats
def r2_func_tf(y_true, y_pred, **kwargs):
    numerator = tf.reduce_sum(tf.square(y_true - y_pred))
    denominator = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - numerator / denominator
    return r2
def rmse_func_tf(y_true, y_pred, **kwargs):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse)
    return rmse
def bias_func_tf(y_true, y_pred, **kwargs):
    bias = tf.reduce_mean(y_true - y_pred)
    return bias
def sdep_func_tf(y_true, y_pred, **kwargs):
    diff = y_true - y_pred
    mean_diff = tf.reduce_mean(diff)
    sdep = tf.sqrt(tf.reduce_mean(tf.square(diff - mean_diff)))
    return sdep

###define CNN archiecture
#CNN model using the k eras.Sequential API
def build_model():
    model = keras.Sequential([
        #first convolution triplet
        layers.Conv1D(32,
                    kernel_size=(3),
                    strides=(2),
                    padding='valid',
                    activation='relu',
                    input_shape=(160,1)),
        layers.MaxPooling1D((2)),
        layers.BatchNormalization(),

        #second convolution triplet
        layers.Conv1D(32,
                    kernel_size=(3),
                    strides=(2),
                    padding='valid',
                    activation='relu'),
        layers.MaxPooling1D((2)),
        layers.BatchNormalization(),

        #third convolution triplet
        layers.Conv1D(32,
                    kernel_size=(3),
                    strides=(2),
                    padding='valid',
                    activation='relu'),
        layers.MaxPooling1D((2)),
        layers.BatchNormalization(),

        #flatten layer 
        layers.Flatten(),

        #4 dense layers
        #the first 3 are hidden layers which have half the number of nodes of the previous layer
        #the 4th layer is the output
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    #setting adam optimiser
    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3)

    #compiling model
    model.compile(loss='mse',
                optimizer=optimiser,
                metrics=['mse', r2_func_tf, rmse_func_tf, bias_func_tf, sdep_func_tf])

    return model

###set global variables
#number of resamples, sfed_type, n_jobs, folds (might be redundant), epochs
n_resamples = 50
sfed_type = 'gf' #possible sfed_types = ['kh', 'hnc', 'gf']
n_jobs = 20
n_folds = 5
epochs = 200

###prepare input data 
input_data = pd.read_csv(f"{datadir}/DATASET")

###separating non-features and features
#defining non-features
meta_cols = ["Mol", "DG", "Temp"]
input_data_meta = input_data[meta_cols]
#defining all features which contains all sfed_types
all_feats = input_data.drop(meta_cols, axis=1, inplace=False)
#creating list of all column heading containing sfed_type
sfed_type_cols = [col for col in all_feats.columns if sfed_type in col]
#creating feature list using only columns corresponding to sfed_type
feats = all_feats[sfed_type_cols]
#defining targets
targets = input_data["DG"]

###create empty lists for resample stats
#training stats
r2_train_resample_stats = []
rmse_train_resample_stats = []
bias_train_resample_stats = []
sdep_train_resample_stats = []
#validation stats
r2_val_resample_stats = []
rmse_val_resample_stats = []
bias_val_resample_stats = []
sdep_val_resample_stats = []
#test stats
r2_test_resample_stats = []
rmse_test_resample_stats = []
bias_test_resample_stats = []
sdep_test_resample_stats = []

###create empty lists for resample predictions
preds_train = []
preds_val = []
preds_test = []

###being resample loop
for resample in range(1, n_resamples+1):

    #beginning timer to determine runtime for each resample
    start_time = time.time()

    print(f"Performing resample {resample} of {n_resamples}...")

    ###create directory for each resample if it does not exist already
    if not pathlib.Path(resultdir/f"resample_{resample}").is_dir():
        pathlib.Path(resultdir/f"resample_{resample}").mkdir()

    #defining output directory for each resample
    outdir = resultdir/f"resample_{resample}"

    ###creates empty lists for CV stats
    #training stats
    r2_train_cv_stats = []
    rmse_train_cv_stats = []
    bias_train_cv_stats = []
    sdep_train_cv_stats = []
    #validation stats
    r2_val_cv_stats = []
    rmse_val_cv_stats = []
    bias_val_cv_stats = []
    sdep_val_cv_stats = []
    #test stats
    r2_test_cv_stats = []
    rmse_test_cv_stats = []
    bias_test_cv_stats = []
    sdep_test_cv_stats = []

    ###create empty lists for CV predictions
    cv_preds_train = []
    cv_preds_val = []
    cv_preds_test = []

    ###random number generator for seed
    rng = random.randrange(0, 2**32-1)

    ###train/test split
    #xtr is training feature, xte is testing features
    xtr = feats.sample(frac=0.7, random_state=rng)
    xte = feats.drop(xtr.index)

    #ytr is targets corresponding to xtr, yte is targets corresponding to xte
    ytr = targets.sample(frac=0.7, random_state=rng)
    yte = targets.drop(ytr.index)
    ytr = ytr.values.squeeze()
    yte = yte.values.squeeze()

    #mtr is meta data corresponding to xtr, mte is meta data corresponding to xte
    mtr = input_data_meta.sample(frac=0.7, random_state=rng)
    mte = input_data_meta.drop(mtr.index)

    ###getting general stats for training data
    xtr_stats = xtr.describe()
    xtr_stats = xtr_stats.transpose()

    ###scaling data
    data_scale = StandardScaler()
    data_scale.fit(xtr)
    nxtr = data_scale.transform(xtr)
    nxte = data_scale.transform(xte)

    ###define callbacks
    keras_callbacks = [
        #early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            start_from_epoch=10, 
            verbose=1, 
            patience=20, 
            restore_best_weights=True
            ),
        #csvlogger to track progress
        tf.keras.callbacks.CSVLogger(
            str(outdir/f"model_history_log_resample_{resample}.csv"), 
            append=False
            ),
        #create logs
        tf.keras.callbacks.TensorBoard(
            log_dir=outdir/"tensorboard_logs", 
            histogram_freq=1, 
            write_graph=False, 
            write_images=False
            ),
    ]

    ###initialise model
    model = KerasRegressor(build_fn=build_model, callbacks=keras_callbacks)

    ###define parameter grid for hyperparameter optimisation
    param_grid = {"batch_size":[16]}

    ###define scoring dict for cv
    scorers = {
        'r2':make_scorer(r2_func), 
        'rmse':make_scorer(rmse_func, greater_is_better=False), 
        'bias':make_scorer(bias_func, greater_is_better=False), 
        'sdep':make_scorer(sdep_func, greater_is_better=False)
        }

    ###create CV using sklearn.GridSearchCV
    grid = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        n_jobs=n_jobs, 
        cv=n_folds, 
        refit='rmse', 
        scoring=scorers, 
        return_train_score=True,
        error_score='raise'
        )

    ###fit the model
    history = grid.fit(nxtr, ytr, 
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2,
                    callbacks=keras_callbacks
                    )

    ###write out cv results and best parameters
    cv_results_df = pd.DataFrame(grid.cv_results_)
    cv_results_df.to_csv(outdir/f"resample_{resample}_cv_results.csv", index=False)
    best_params = str(grid.best_params_)
    with open(outdir/'best_params.txt', 'w+') as f:
        f.write(best_params)

    #identify best hyperparameters
    best_params = grid.best_params_

    #make predictions on testing split
    preds = history.predict(nxte)
    preds = preds.squeeze()

    #define meta data for predicted samples
    mte["Prediction"] = preds

    #append test prediction to previously defined list
    preds_test.append(preds)

    #writing output 
    mte.to_csv(outdir/"test_set_preds.csv", index=False)

    ###calculating stats per resamples
    r2_test = metrics.r2_score(yte, preds)
    rmse_test = np.sqrt(metrics.mean_squared_error(yte, preds))
    bias_test = np.mean(yte-preds)
    sdep_test = (np.mean((yte-preds-bias_test)**2))**0.5

    ###appending calculated stats to lists defined earlier
    r2_test_resample_stats.append(r2_test)
    rmse_test_resample_stats.append(rmse_test)
    bias_test_resample_stats.append(bias_test)
    sdep_test_resample_stats.append(sdep_test)

    #stopping timer
    end_time = time.time()

    #determining runtime for resample
    time_taken = end_time - start_time

    print(f"Time taken to complete resample_{resample} was {time_taken} seconds.")

###formatting stats to output overall model stats
resample_stats_df = pd.DataFrame({
    'r2':r2_test_resample_stats,
    'rmse':rmse_test_resample_stats,
    'bias':bias_test_resample_stats,
    'sdep':sdep_test_resample_stats
    })
    
#calculating mean and standard deviations of calculated stats
resample_stats_summary_df = pd.DataFrame([
    resample_stats_df.mean().to_dict(),
    resample_stats_df.std().to_dict()
], index=['mean','std'])

#writing out individual stats for each resample into a single csv
resample_stats_df.to_csv(resultdir/"resample_stats.csv", index=False)
#writing out the mean and standard deviation values to a separate csv
resample_stats_summary_df.to_csv(resultdir/"resample_stats_summary.csv")    







