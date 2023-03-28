import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


def get_data(
    data_path: str,                       # file path of data to read
    train_sample_num: float = 450,        # number of samples to use for training
    test_sample_num: float = 67,          # number of samples to use for testing
    is_bias: bool = True,                 # whether to add a bias term to the features
    is_shuffle: bool = False,             # whether to shuffle the samples
    )-> tuple:

    X_list = []                           # create an empty list to store the features
    Y_list = []                           # create an empty list to store the labels
    df = pd.read_csv(data_path)           # read the data from the specified file
    for index in df.index:                # loop over each row in the data frame
        X_piece = np.array(df.loc[index,['X','Y','1th','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain',]]) # extract the features as a numpy array
        X_list.append(X_piece)            # append the features to the list
        Y_piece = int(df.loc[index,['area']])   # extract the label
        Y_list.append(Y_piece)            # append the label to the list

    X = np.array(X_list)                  # convert the list of features to a numpy array
    if is_bias:                           # if the bias flag is set to True
        X = np.concatenate([np.ones((len(X_list),1)),X],axis=1).astype(np.float32)  # add a column of ones to the features as the bias term
    else:                                 # if the bias flag is set to False
        X = X.astype(np.float32)          # cast the features as a float32 numpy array
    Y = np.array(Y_list,dtype=np.float32) # convert the list of labels to a float32 numpy array
    Y = np.log(Y+1)                       # apply logarithmic transformation to the labels

    #shuffle
    if is_shuffle:                        # if the shuffle flag is set to True
        indexes = np.arange(0,len(X),1,dtype=np.int32)   # create an array of indexes for the samples
        np.random.shuffle(indexes)        # shuffle the indexes randomly
        X = X[indexes]                    # shuffle the features based on the shuffled indexes
        Y = Y[indexes]                    # shuffle the labels based on the shuffled indexes

    X_train = X[:train_sample_num]        # extract the training features from the beginning of the numpy array
    Y_train = Y[:train_sample_num]        # extract the training labels from the beginning of the numpy array
    X_test = X[-test_sample_num:]          # extract the testing features from the end of the numpy array
    Y_test = Y[-test_sample_num:]          # extract the testing labels from the end of the numpy array

    return X_train,X_test,Y_train,Y_test  # return the training and testing features and labels as a tuple
