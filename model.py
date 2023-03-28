# This file is implemention for linear model, Ridge Regression, RBF kernel regression,
#  Spline Regression and Lasso Regression.
import numpy as np
from dataset import get_data
from data_preprocess import DataSet
from matplotlib import pyplot as plt

# y_i = beta.T @ X_i
class LinearRegression:
    def __init__(self, X, Y, seed = 123) -> None:
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
        m, n = X.shape
        self.sigma = sigma
        self.Lambda = Lambda
        self.num = len(Y)
        self.c = np.random.random(m)
        self.K = self.__Cal_K__(self.X, self.X)
        
    def __Cal_K__(self, X, Y):
        n, p = X.shape
        m, p = Y.shape
        X_2 = (np.sum(X**2, axis = 1) * np.ones((m, n))).T
        X_Y = X @ Y.T
        Y_2 = (np.sum(Y**2, axis = 1) * np.ones((n, m)))
        t = X_2 - 2 * X_Y + Y_2
        return np.exp(-t / (2 * self.sigma**2)).T
    
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
        self.c = np.linalg.inv(self.K + self.Lambda * np.eye(len(self.c))) @ self.Y
    
    def update(self, lr = 0.000002):
        self.c = self.c - lr * self.deri
    
    def predict(self, X_): 
        K = self.__Cal_K__(self.X, X_)
        return K @ self.c

#l(beta) = 1/2 * (Y - X @ beta).T @ (Y - X @ beta) + Lambda * |beta|
class LassoRegression:
    def __init__(self, X, Y, Lambda = 0.01, seed = 123) -> None:
        np.random.seed(42)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Lambda = Lambda
        self.num = len(Y)
        self.beta = np.random.random(X.shape[1])
        
    def loss(self):
        Y_predict = self.predict(self.X)
        self.loss_ = (0.5 * (self.Y - Y_predict).T @ (self.Y - Y_predict) + \
            self.Lambda * np.linalg.norm(x= self.beta, ord= 1)) / self.num
        return self.loss_
    
    def derivative(self):
        Y_predict = self.predict(self.X)
        self.deri = (self.X.T @ (Y_predict - self.Y) + self.Lambda * np.sign(self.beta)) / self.num
        return self.deri
    
    def update(self, lr = 0.000002):
        self.beta = self.beta - lr * self.deri
    
    def predict(self, X): 
        return X @ self.beta
    
    def CoordinateDescent(self, a, b, interval = 100):
        n, p = self.X.shape
        delta = (b - a) / interval
        for t in range(interval):
            Lambda = 10**(b - t * delta)
            for j in range(p):
                beta = self.beta.copy()
                beta[j] = 0
                R = self.Y - self.X @ beta
                Xj = self.X.T[j]
                gamma = R @ Xj / (Xj.T @ Xj)
                self.beta[j] = np.sign(gamma) * max(0, abs(gamma) - Lambda / (Xj.T @ Xj))
    
def sse_loss(Y_predict, Y_test):
    N = len(Y_test)
    sum = 0
    for (i, j) in zip(Y_predict, Y_test):
        sum += (i - j)**2
    return sum

def polt_beta(beta, name):
    x = beta.copy()
    x = abs(np.sort(-x))
    plt.bar(range(13), x)
    plt.grid(True,linestyle=':',color='r',alpha=0.6)
    plt.title('Lasso Beta')
    plt.savefig(f'fig/{name}.png')

if __name__ == "__main__":
    epochs = 1000000
    X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")
    Data = DataSet(X_train, X_test, Y_train, Y_test)
    Data.normalize() # Data normalization
    X_train, Y_train = Data.train_set()
    X_test, Y_test = Data.test_set()
    
    # Linear = LinearRegression(X_train, Y_train)
    Ridge = RidgeRegression(X_train, Y_train, 90)
    Ridge.analysis_fit(X_train, Y_train)
    
    Kernel = RBFKernelRegression(X_train, Y_train, 5, 50)
    Kernel.analysis_fit()
    
    Lasso = LassoRegression(X_train, Y_train, 60)
    Lasso.CoordinateDescent(1.5, 0.01, 1000)
    # for epoch in range(epochs):
    #     loss = Linear.loss()
    #     deri = Linear.derivative()
    #     Linear.update()
    #     if(epoch%5000 == 0):
    #         print(f'Epoch {epoch}: loss = {loss}, derivative = {deri}')
    
    # for epoch in range(epochs):
    #     loss = Lasso.loss()
    #     deri = Lasso.derivative()
    #     Lasso.update()
    #     if(epoch%5000 == 0):
    #         print(f'Epoch {epoch}: loss = {loss}, derivative = {deri}')
            
    # print(f'Linear: {sse_loss(Linear.predict(X_test), Y_test)}')
    # print(f'Ridge: {sse_loss(Ridge.predict(X_test), Y_test)}')
    # print(f'RBFKerner: {sse_loss(Kernel.predict(X_test), Y_test)}')

    print(f'Lasso: {sse_loss(Lasso.predict(X_test), Y_test)}')
    print(Lasso.beta)
    polt_beta(Lasso.beta, 'Lasso')