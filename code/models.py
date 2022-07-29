from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, ReLU, Dropout, Activation, Concatenate, Reshape, MaxPooling3D
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Bidirectional, BatchNormalization, Flatten, Input, GRU
from tensorflow.keras.layers import ConvLSTM3D, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K
from config import SVM_SETTINGS, LSTM_SETTINGS, CATBOOST_SETTINGS, CNN_SETTINGS, ENSAMBLE_SETTINGS, RF_SETTINGS
#from configPeru import SVM_SETTINGS, LSTM_SETTINGS, CATBOOST_SETTINGS, CNN_SETTINGS, ENSAMBLE_SETTINGS, RF_SETTINGS
from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
import itertools
import warnings
import pickle

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


class CNN:
    def __init__(self, shape):
        self.shape = shape
        self.epochs = CNN_SETTINGS['EPOCHS']
        self.lr = CNN_SETTINGS['LEARNING RATE']
        self.batch_size = CNN_SETTINGS['BATCH SIZE']
        self.loss = CNN_SETTINGS['LOSS']
        self.eval_metric = CNN_SETTINGS['EVALUATION METRIC']
        self.early_stopping_rounds = CNN_SETTINGS['EARLY STOPPING']
        self.optimizer = Adam(learning_rate=self.lr)#CNN_SETTINGS['OPTIMZER']

        self.model = self.__build()

    def __build(self):
        # shape -> (batches, time, width, height, variables)
        input_layer = Input(self.shape)

        #x = ConvLSTM3D(
        #  16, kernel_size = (7,7,7), strides = (3,3,3), 
        # data_format='channels_last', activation='tanh',
        # recurrent_activation='hard_sigmoid', use_bias=True,
        # kernel_initializer='glorot_uniform',
        # return_sequences=True, 
        #)(input_layer)

        x = ConvLSTM2D(20, 
          kernel_size=(7,7), 
          strides=(3,3), 
          data_format='channels_last', 
          return_sequences=True)(input_layer)

        x = ConvLSTM2D(1, 
          kernel_size=(7,7), 
          strides=(3,3), 
          data_format='channels_last', 
          return_sequences=False)(input_layer)

        #x = ConvLSTM3D(
        # 4, kernel_size = (7,7,7), strides = (3,3,3),  
        # activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
        #kernel_initializer='glorot_uniform',
        # return_sequences=False, 
        #)(x)

        x = Flatten()(x)
        x = Dense(27*2)(x)

        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss=self.loss, metrics = self.eval_metric, optimizer=self.optimizer)
        return model


class Ensamble:
    def __init__(self, shape):
        self.shape = shape
        self.epochs = ENSAMBLE_SETTINGS['EPOCHS']
        self.lr = ENSAMBLE_SETTINGS['LEARNING RATE']
        self.batch_size = ENSAMBLE_SETTINGS['BATCH SIZE']
        self.loss = ENSAMBLE_SETTINGS['LOSS']
        self.eval_metric = ENSAMBLE_SETTINGS['EVALUATION METRIC']
        self.early_stopping_rounds = ENSAMBLE_SETTINGS['EARLY STOPPING']
        if ENSAMBLE_SETTINGS['EARLY STOPPING'] == 'adam':
            self.optimizer = Adam(learning_rate = self.lr)
        elif ENSAMBLE_SETTINGS['EARLY STOPPING'] == 'rmsprop':
            self.optimizer = ENSAMBLE_SETTINGS['EARLY STOPPING']
        else:
            self.optimizer = 'adam'

        self.model = self.__build()

    def load(self, path):
        self.model = load_model(path)

    def __build(self):
        catboost_in = Input(shape=self.shape)
        svm_in = Input(shape=self.shape)
        lstm_in = Input(shape=self.shape)
        
        catboostx1 = Dense(16)(catboost_in)
        catboostx1 = Activation('relu')(catboostx1)

        sv1_x1 = Dense(16)(svm_in)
        sv1_x1 = Activation('relu')(sv1_x1)

        lstm_x1 = Dense(16)(lstm_in)
        lstm_x1 = Activation('relu')(lstm_x1)

        x = Concatenate()([catboostx1, sv1_x1, lstm_x1])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(self.shape)(x)
        x = Activation('relu')(x)

        model = Model(inputs = [catboost_in, svm_in, lstm_in], outputs=x)
        model.compile(loss=self.loss, metrics=self.eval_metric, optimizer=self.optimizer)
        return model

    def train(self, x_train, y_train, x_val, y_val, output_path):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.early_stopping_rounds,
                           verbose=0, mode='auto', baseline=None, restore_best_weights=False)

        history = self.model.fit(
            x = x_train,
            y = y_train,
            validation_data = (x_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[es],
            shuffle = True
        )

        today = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        self.model.save(os.path.join(output_path,'ENSEMBLE-'+today+'.h5'))
        return history


class CatBoostEnsableNet:
    def __init__(self):
        self.epochs = ENSAMBLE_SETTINGS['EPOCHS']
        self.device = CATBOOST_SETTINGS['DEVICE']
        self.lr = ENSAMBLE_SETTINGS['LEARNING RATE']
        self.loss = CATBOOST_SETTINGS['LOSS']
        self.seed = CATBOOST_SETTINGS['RANDOM SEED']
        self.max_depth = CATBOOST_SETTINGS['MAX DEPTH']
        self.eval_metric = CATBOOST_SETTINGS['EVALUATION METRIC']
        self.early_stopping_rounds = CATBOOST_SETTINGS['EARLY STOPPING']

        self.model = self.__build()

    def __build(self):
        return CatBoostRegressor(
            iterations=self.epochs,
            task_type=self.device,
            learning_rate=self.lr,
            loss_function=self.loss,  #MAE, RMSE, MultiRMSE for multi-regression
            random_seed=self.seed,
            max_depth=self.max_depth,
            eval_metric=self.eval_metric,  #MAE, RMSE, MultiRMSE for multi-regression
            verbose = True,
            early_stopping_rounds = self.early_stopping_rounds
        )

    def load(self, path):
        regressor = CatBoostRegressor()
        regressor.load_model(path)
        self.model = regressor

    def train(self, x_train, y_train, x_val, y_val, output_path):
        train_pool = Pool(x_train, y_train)
        val_pool = Pool(x_val, y_val)
        self.model.fit(train_pool,eval_set=val_pool)

        today = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        self.model.save_model(os.path.join(output_path,'ENSEMBLE-'+today))


class RandomForestEnsableNet:
    def __init__(self, finetuning):
        self.n_estimators = RF_SETTINGS['NB_ESTIMATORS']
        self.max_depth = RF_SETTINGS['MAX DEPTH']
        self.finetuning = finetuning
        self.model = self.__build()
        

    def __build(self):
        return RandomForestRegressor(
            n_estimators = self.n_estimators,
            #max_depth=self.max_depth,
            warm_start=self.finetuning
        )

    def load(self, path):
        with open(path,'rb') as f:
            regressor = pickle.load(f)
        self.model = regressor

    def train(self, x_train, y_train, x_val, y_val, output_path):
        self.model.fit(x_train, y_train)

        today = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        with open(os.path.join(output_path,'RF-'+today+'.pkl'),'wb') as f:
            pickle.dump(self.model, f)


class LSTMNet:
    def __init__(self, shape):
        self.shape = shape
        self.epochs = LSTM_SETTINGS['EPOCHS']
        self.lr = LSTM_SETTINGS['LEARNING RATE']
        self.batch_size = LSTM_SETTINGS['BATCH SIZE']
        self.loss = LSTM_SETTINGS['LOSS']
        self.eval_metric = LSTM_SETTINGS['EVALUATION METRIC']
        self.early_stopping_rounds = LSTM_SETTINGS['EARLY STOPPING']

        if LSTM_SETTINGS['EARLY STOPPING'] == 'adam':
            self.optimizer = Adam(learning_rate = self.lr)
        elif LSTM_SETTINGS['EARLY STOPPING'] == 'rmsprop':
            self.optimizer = LSTM_SETTINGS['EARLY STOPPING']
        else:
            self.optimizer = 'adam'

        self.model = self.__build()

    def __build(self):
        model = Sequential()
        # Adding a Bidirectional LSTM layer
        model.add(LSTM(60, return_sequences=True, dropout=0.5, input_shape=self.shape))
        model.add(LSTM(20, dropout=0.5))
        model.add(Dense(2))
        model.compile(loss=self.loss, metrics = self.eval_metric, optimizer=self.optimizer)
        return model

    def load(self, path):
        self.model = load_model(path)

    def train(self, training, validation, output_path):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.early_stopping_rounds, 
                           verbose=0, mode='auto', baseline=None, restore_best_weights=False)

        history = self.model.fit(
            x = training[0],
            y = training[1],
            validation_data = (validation[0], validation[1]),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[es],
            shuffle = True
        )

        today = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        self.model.save(os.path.join(output_path,'LSTM-'+today+'.h5'))
        return history


class CatBoostNet:
    def __init__(self):
        self.epochs = CATBOOST_SETTINGS['EPOCHS']
        self.device = CATBOOST_SETTINGS['DEVICE']
        self.lr = CATBOOST_SETTINGS['LEARNING RATE']
        self.loss = CATBOOST_SETTINGS['LOSS']
        self.seed = CATBOOST_SETTINGS['RANDOM SEED']
        self.max_depth = CATBOOST_SETTINGS['MAX DEPTH']
        self.eval_metric = CATBOOST_SETTINGS['EVALUATION METRIC']
        self.early_stopping_rounds = CATBOOST_SETTINGS['EARLY STOPPING']

        self.model = self.__build()

    def __build(self):
        return CatBoostRegressor(
            iterations=self.epochs,
            task_type=self.device,
            learning_rate=self.lr,
            loss_function=self.loss,  #MAE, RMSE, MultiRMSE for multi-regression
            random_seed=self.seed,
            max_depth=self.max_depth,
            eval_metric=self.eval_metric,  #MAE, RMSE, MultiRMSE for multi-regression
            verbose = True,
            early_stopping_rounds = self.early_stopping_rounds
        )

    def load(self, path):
        regressor = CatBoostRegressor()
        regressor.load_model(path)
        self.model = regressor

    def train(self, training, validation, output_path):
        train_pool = Pool(training[0], training[1])
        val_pool = Pool(validation[0], validation[1])
        self.model.fit(train_pool,eval_set=val_pool)

        today = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        self.model.save_model(os.path.join(output_path,'CATBOOST-'+today))


class SVMNet:
    def __init__(self):
        self.epochs = SVM_SETTINGS['EPOCHS']
        self.random_state = SVM_SETTINGS['RANDOM STATE']
        self.n_iter = SVM_SETTINGS['N ITER']
        self.cv = SVM_SETTINGS['CV']
        self.hyperparameters = SVM_SETTINGS['HYPERPARAM']
      
        self.model, self.randomized_search = self.__build()

    def __build(self):
        model = MultiOutputRegressor(SVR(max_iter=self.epochs, verbose=True))
        randomized_search = RandomizedSearchCV(
            model, self.hyperparameters, 
            random_state=self.random_state, 
            n_iter=self.n_iter, 
            scoring=None,
            refit=True, 
            cv=self.cv, 
            verbose=True, 
            error_score='raise', 
            return_train_score=True
            )
        return model, randomized_search

    def load(self, path):
        with open(path,'rb') as f:
            regressor = pickle.load(f)
      
        self.model = regressor

    def train(self, training, validation, output_path):
        hyperparameters_tuning = self.randomized_search.fit(training[0], training[1])
        tuned_model = hyperparameters_tuning.best_estimator_
        self.model = tuned_model

        today = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

        with open(os.path.join(output_path,'SVM-'+today+'.pkl'),'wb') as f:
            pickle.dump(tuned_model, f)
