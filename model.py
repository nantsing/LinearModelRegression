# This file is implemention for linear model, Ridge Regression, RBF kernel regression,
#  Spline Regression and Lasso Regression.
import numpy as np
from dataset import get_data
# y_i = beta^T * X_i
class LinearRegression:
    def __init__(self, X, Y) -> None:
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
    

class RidgeRegression:
    def __init__(self, X, Y, Lambda = 0.01) -> None:
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
    
class RBFKernelRegression:
    pass

class LassRegression:
    pass

def sse_loss(Y_predict, Y_test):
    N = len(Y_test)
    sum = 0
    for (i, j) in zip(Y_predict, Y_test):
        sum += (i - j)**2
    return sum

if __name__ == "__main__":
    epochs = 10000000
    X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")
    Linear = LinearRegression(X_train, Y_train)
    Ridge = RidgeRegression(X_train, Y_train, 100000)
    Linear.analysis_fit(X_train, Y_train)
    Ridge.analysis_fit(X_train, Y_train)
    # for epoch in range(epochs):
    #     loss = Linear.loss()
    #     deri = Linear.derivative()
    #     Linear.update()
    #     if(epoch%5000 == 0):
    #         print(f'Epoch {epoch}: loss = {loss}, derivative = {deri}')
            
    # print(sse_loss(model.predict(X_test), Y_test))
    print(sse_loss(Linear.predict(X_test), Y_test))
    print(sse_loss(Ridge.predict(X_test), Y_test))