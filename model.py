# This file is implemention for linear model, Ridge Regression, RBF kernel regression,
#  Spline Regression and Lasso Regression.
import numpy as np
from dataset import get_data
from data_preprocess import DataSet

# y_i = beta.T @ X_i
class LinearRegression:
    def __init__(self, X, Y, seed = 42) -> None:
        np.random.seed(42)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.num = len(Y)
        self.beta = np.random.random(X.shape[1])
        
    def loss(self):
        Y_predict = self.predict(self.X)
        self.loss_ = (self.Y - Y_predict).T @ (self.Y - Y_predict) / self.num
        return self.loss_
    
    def derivative(self):
        Y_predict = self.predict(self.X)
        self.deri = 2 * self.X.T @ (Y_predict - self.Y) / self.num
        return self.deri
        
    def analysis_fit(self, X_train, Y_train):
        X = np.array(X_train)
        Y = np.array(Y_train)
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    
    def update(self, lr = 0.000002):
        self.beta = self.beta - lr * self.deri
    
    def predict(self, X):
        return X @ self.beta
    

#l(beta) = (Y - beta.T @ X).T @ (Y - beta.T @ X) + Lambda * beta.T @ beta
class RidgeRegression:
    def __init__(self, X, Y, Lambda = 0.01, seed = 42) -> None:
        np.random.seed(42)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Lambda = Lambda
        self.num = len(Y)
        self.beta = np.random.random(X.shape[1])
        
    def loss(self):
        Y_predict = self.predict(self.X)
        self.loss_ = ((self.Y - Y_predict).T @ (self.Y - Y_predict) + \
            self.Lambda * self.beta.T @ self.beta) / self.num
        return self.loss_
    
    def derivative(self):
        Y_predict = self.predict(self.X)
        self.deri = (2 * self.X.T @ (Y_predict - self.Y) + 2 * self.Lambda * self.beta) / self.num
        return self.deri
        
    def analysis_fit(self, X_train, Y_train):
        X = np.array(X_train)
        Y = np.array(Y_train)
        self.beta = np.linalg.inv(X.T @ X + self.Lambda * np.eye(len(self.beta))) @ X.T @ Y
    
    def update(self, lr = 0.000002):
        self.beta = self.beta - lr * self.deri
    
    def predict(self, X): 
        return X @ self.beta
    
#l(c) = (Y - Kc).T @ (Y - Kc) + Lambda * c.T @ K @ c
class RBFKernelRegression:
    def __init__(self, X, Y, Lambda = 0.01, sigma = 0.1, seed = 42) -> None:
        np.random.seed(42)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.sigma = sigma
        self.Lambda = Lambda
        self.num = len(Y)
        self.c = np.random.random(X.shape[0])
        m, n = self.X.shape
        X_2 = (np.sum(self.X**2, axis = 1) * np.ones((m, m)))
        X_Y = self.X @ self.X.T
        t = X_2 - 2 * X_Y + X_2.T
        self.K = np.exp(-t / (2 * self.sigma**2))
    
    def loss(self):
        Y_predict = self.predict(self.X)
        self.loss_ = ((self.Y - Y_predict).T @ (self.Y - Y_predict) + \
            self.Lambda * self.c.T @ self.K @ self.c) / self.num
        return self.loss_
    
    def derivative(self):
        Y_predict = self.predict(self.X)
        self.deri = (2 * self.X.T @ (Y_predict - self.Y) + 2 * self.Lambda * self.K @ self.c) / self.num
        return self.deri
        
    def analysis_fit(self):
        self.beta = np.linalg.inv(self.K + self.Lambda * np.eye(len(self.c))) @ Y
    
    def update(self, lr = 0.000002):
        self.c = self.c - lr * self.deri
    
    def predict(self, X): 
        return K @ self.c

class LassRegression:
    pass

def sse_loss(Y_predict, Y_test):
    N = len(Y_test)
    sum = 0
    for (i, j) in zip(Y_predict, Y_test):
        sum += (i - j)**2
    return sum

if __name__ == "__main__":
    epochs = 1000000
    X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")
    Data = DataSet(X_train, X_test, Y_train, Y_test)
    Data.normalize() # Data normalization
    X_train, Y_train = Data.train_set()
    X_test, Y_test = Data.test_set()
    Kernel = RBFKernelRegression(X_train, Y_train)
    # Linear = LinearRegression(X_train, Y_train)
    # Ridge = RidgeRegression(X_train, Y_train, 90)
    # Ridge.analysis_fit(X_train, Y_train)
    # for epoch in range(epochs):
    #     loss = Linear.loss()
    #     deri = Linear.derivative()
    #     Linear.update()
    #     if(epoch%5000 == 0):
    #         print(f'Epoch {epoch}: loss = {loss}, derivative = {deri}')
            
    # print(sse_loss(Linear.predict(X_test), Y_test))
    # print(sse_loss(Ridge.predict(X_test), Y_test))