# This file is implemention for linear model, Ridge Regression, RBF kernel regression,
#  Spline Regression and Lasso Regression.
import numpy as np
from dataset import get_data
# from data_preprocess import DataSet
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

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
    
    def update(self, lr = 0.0000002):
        self.beta = self.beta - lr * self.deri
    
    def predict(self, X):
        return X @ self.beta

    def load(self, path):
        self.beta = np.load(path)
    

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
    
    def update(self, lr = 0.00033):
        self.beta = self.beta - lr * self.deri
    
    def predict(self, X): 
        return X @ self.beta
    
    def load(self, path):
        self.beta = np.load(path)
    
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
    
    def update(self, lr = 0.0000001):
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
    
    def update(self, lr = 0.001):
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

    def load(self, path):
        self.beta = np.load(path)
    
def sse_loss(Y_predict, Y_test):
    N = len(Y_test)
    sum = 0
    for (i, j) in zip(Y_predict, Y_test):
        sum += (i - j)**2
    return sum

def polt_beta(beta, name, is_removed_bias = 1):
    x = beta.copy()
    attri = np.array(['bias','X','Y','1th','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain'])
    if is_removed_bias:
        attri = attri[1:]
        x = x[1:]

    colors = np.where(x > 0, 'r', 'b')
    x = abs(x)
    adj_idx = np.argsort(-x)
    
    plt.figure()
    plt.bar(attri[adj_idx], x[adj_idx], color = colors[adj_idx], alpha = 0.6)
    plt.grid(True, linestyle=':', color='r', alpha=0.6)
    pos_patch = mpatches.Patch(color='red', label='Positive', alpha = 0.6)
    neg_patch = mpatches.Patch(color='blue', label='Negative', alpha = 0.6)
    plt.legend(handles=[pos_patch, neg_patch])
    plt.title(f'{name} Beta')
    plt.savefig(f'fig/{name}.png')

#### to test ####
def plot_multiBeta(Beta, Labelist, figname, is_removed_bias = 1):
    Y = Beta.copy()
    attri = np.array(['bias','X','Y','1th','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain'])
    if is_removed_bias:
        attri = attri[1:]
        Y = Y.T[1:].T
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure()
    
    for idx, (beta, label) in enumerate(zip(Beta, Labelist)):
        plt.plot(attri, beta, colors[idx], marker = 'o', alpha = 0.6, label = Labelist[idx])
        
    plt.legend()
    plt.title(f'{figname} MultiBeta')
    plt.savefig(f'fig/{figname}.png')
    

def plot_c(c, name):
    x = c.copy()
    colors = np.where(x > 0, 'r', 'b')
    x = abs(x)
    adj_idx = np.argsort(-x)
    plt.figure()
    plt.bar(range(len(c)), x[adj_idx], color = colors[adj_idx], alpha = 0.6)
    plt.xlim((0, len(c)))
    plt.ylim((0, x.max()))
    plt.grid(True, linestyle=':', color='r', alpha=0.6)
    pos_patch = mpatches.Patch(color='red', label='Positive', alpha = 0.6)
    neg_patch = mpatches.Patch(color='blue', label='Negative', alpha = 0.6)
    plt.legend(handles=[pos_patch, neg_patch])
    plt.title(f'{name} c')
    plt.savefig(f'fig/{name}.png')

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")
    
    Linear = LinearRegression(X_train, Y_train)
    Linear.analysis_fit(X_train, Y_train)
    print(Linear.beta)
    
    Ridge = RidgeRegression(X_train, Y_train, 90)
    Ridge.analysis_fit(X_train, Y_train)
    print(Ridge.beta)
    
    Kernel = RBFKernelRegression(X_train, Y_train, 5, 50)
    Kernel.analysis_fit()
    print(f'RBFKerner: {sse_loss(Kernel.predict(X_test), Y_test)}')

    Lasso = LassoRegression(X_train, Y_train, 60)
    Lasso.CoordinateDescent()
    print(Lasso.beta)