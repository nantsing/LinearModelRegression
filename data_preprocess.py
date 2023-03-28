#Some preprocess for raw data including normalization
import numpy as np

class DataSet:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.norm = False

    def normalize(self):
        self.norm = True
        self.X_train_norm = self.X_train / np.max(self.X_train, axis = 0)
        self.X_test_norm = self.X_test / np.max(self.X_train, axis = 0)
        
    def train_set(self):
        if self.norm:
            return self.X_train_norm, self.Y_train
        else:
            return self.X_train, self.Y_train
    
    def test_set(self):
        if self.norm:
            return self.X_test_norm, self.Y_test
        else:
            return self.X_test, self.Y_test

    def raw_data(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test
