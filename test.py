# For calculate correlation coefficient and determination coefficient
import math
import numpy as np
from model import *
from dataset import get_data

X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")

def computeCorrelation(X, Y):
    Cov = np.cov(X, Y)
    Var_X = Cov[0, 0]
    Var_Y = Cov[1, 1]
    Cov_XY = Cov[0, 1]
    return Cov_XY / math.sqrt(Var_X * Var_Y)
    
def polyfit(X, Y, path = None):
    n, p = X.shape
    Linear = LinearRegression(X, Y)
    if path != None: Linear.load(path)
    else: Linear.analysis_fit(X, Y)
    Y_predict = Linear.predict(X)
    Y_mean = Y.mean()
    SSR = np.linalg.norm(Y_predict - Y_mean, ord= 2)**2
    SST = np.linalg.norm(Y - Y_mean, ord= 2)**2
    return SSR / SST * ((n - 1) / (n - p))


if __name__ == '__main__':
    attris = ['X','Y','1th','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain']
    print("Correlation coefficient")
    for (attri, X) in zip(attris, X_train.T[1:]):
        linearCorrelation = computeCorrelation(X, Y_train)
        print(f'{attri}: {linearCorrelation}.')
    print()
    print("Determination coefficient")
    print(polyfit(X_train, Y_train))
    print(polyfit(X_train, Y_train, './models/Linear.npy'))
        