import pandas as pd
import numpy as np


class datasetHandler:

    def __init__(self, train_dataframe, val_dataframe):
        self.training = train_dataframe
        self.validation = val_dataframe

    def get_data(self, t_window_size, t_prediction):

        # Get the first department to asses data dimensions
        departments = pd.unique(self.training.dep_id)
        db_dep = self.training[self.training.dep_id == departments[0]]
        # len_series = db_dep.shape[0] - (t_window_size + t_prediction)
        len_series = db_dep.shape[0] - (t_window_size)

        # Initialize training dataset
        x_train = np.zeros((len_series*len(departments), t_window_size, self.training.shape[1]))
        y_train = np.zeros((len_series*len(departments), 2*t_prediction))
        print('X Training shape', x_train.shape)
        print('Y Training shape', x_train.shape)

        # Fill the training set
        counter = 0
        for count, dep in enumerate(departments):
            db_dep = self.training[self.training.dep_id == dep]
            data = db_dep.to_numpy()

            print('\rProcessing departments {} of {}'.format(count, len(departments)), end='\t\t')

            for count2 in range(len_series):
                x_train[counter, ...] = data[count2 : (count2)+t_window_size, :]
                y_train[counter, ..., :t_prediction] = data[(count2 + t_window_size) : (count2 + t_window_size + t_prediction), -2]
                y_train[counter, ..., t_prediction:] = data[(count2 + t_window_size) : (count2 + t_window_size + t_prediction), -1]
                counter+=1

        #  Get the first department to asses data dimensions
        departments = pd.unique(self.validation.dep_id)
        db_dep = self.validation[self.validation.dep_id == departments[0]]
        # len_series  = db_dep.shape[0] - (t_window_size + t_prediction)
        len_series = db_dep.shape[0] - (t_window_size)
        
        # Initialize validation dataset
        x_val = np.zeros((len_series*len(departments), t_window_size, self.validation.shape[1]))
        y_val = np.zeros((len_series*len(departments), 2*t_prediction))
        print('\nX Validation shape', x_val.shape)
        print('Y Validation shape', y_val.shape)

        counter = 0
        # Fill the validation set
        for count, dep in enumerate(departments):
            db_dep = self.validation[self.validation.dep_id == dep]
            data = db_dep.to_numpy()

            print('\rProcessing departments {} of {}'.format(count, len(departments)), end='\t\t')

            for count2 in range(len_series):
                x_val[counter, ...] = data[count2 : (count2)+t_window_size, :]
                y_val[counter, ..., :t_prediction] = data[count2 + t_window_size:count2 + t_window_size + t_prediction, -2]
                y_val[counter, ..., t_prediction:] = data[count2 + t_window_size:count2 + t_window_size + t_prediction, -1]
                counter += 1

        return x_train, y_train, x_val, y_val

    def augment(self, x_train, y_train, x_val, y_val, multiplier = 3):
        x_train_a = np.zeros((multiplier*x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        y_train_a = np.zeros((multiplier*y_train.shape[0], y_train.shape[1]))
        x_val_a   = np.zeros((multiplier*x_val.shape[0],   x_val.shape[1],   x_val.shape[2]))
        y_val_a   = np.zeros((multiplier*y_val.shape[0],   y_val.shape[1]))

        for i in range(multiplier):
            x_train_a[i*x_train.shape[0]:(i+1)*x_train.shape[0], ...] = x_train + np.random.normal(0, 0.01, size =x_train.shape[0]*x_train.shape[1]*x_train.shape[2]).reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])
            x_val_a[i*x_val.shape[0]:(i+1)*x_val.shape[0], ...] = x_val + np.random.normal(0, 0.01, size =x_val.shape[0]*x_val.shape[1]*x_val.shape[2]).reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2])
            y_train_a[i*y_train.shape[0]:(i+1)*y_train.shape[0], ...] = y_train
            y_val_a[i*y_val.shape[0]:(i+1)*y_val.shape[0], ...] = y_val

        return x_train_a, y_train_a, x_val_a, y_val_a

    def prepare_data_LSTM(self, x_train, y_train, x_val, y_val):
        return (x_train, y_train), (x_val, y_val)

    def prepare_data_CatBoost(self, x_train, y_train, x_val, y_val):
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
        x_val   =   x_val.reshape((x_val.shape[0],   x_val.shape[1]*x_val.shape[2]))
        return (x_train, y_train), (x_val, y_val)
