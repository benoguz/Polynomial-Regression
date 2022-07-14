from sklearn import preprocessing
import numpy as np


class PolyRegNp():
    def __init__(self, X, y):
        super().__init__()
        self.X = X.reshape(-1,1)
        self.y = y.reshape(-1,1)
        self.X_orig_shape = X.shape
        self.y_orig_shape = y.shape
        self.X_scaled = None
        self.coef = None

    def feature_scaling(self):
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.X,self.y)
        self.X_scaled = scaler.transform(self.X)
        return self.X_scaled, self.y
    
    def model_compute(self):
        degree = len(self.coef)-1
        curve = [np.sum([self.coef[-1]] + [x**(degree-d)*c for d,c \
                in enumerate(self.coef[:-1])]) for x in self.X_scaled]
        return curve

    def fit(self, degree = 4):
        self.coef = np.polyfit(self.X_scaled.flatten(), self.y, degree)
        return self.coef
    
    def predict(self):
        y_pred = self.model_compute()
        return y_pred
        
import torch
import torch.nn as nn

class PolyRegTorch():
    def __init__(self, X, y, standardize_nd_features = False):
        super().__init__()
        self.X = X
        self.y = y
        self.X_orig_shape = X.shape
        self.y_orig_shape = y.shape
        self.X_scaled = None
        self.standardize_nd_features = standardize_nd_features
        
    def feature_scaling(self):
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.X.reshape(-1,1),self.y)
        self.X_scaled = scaler.transform(self.X.reshape(-1,1))
        self.X_scaled = torch.from_numpy(self.X_scaled).float()
        self.y = torch.from_numpy(self.y).float()
        return self.X_scaled , self.y
    
    def n_degree_features(self, degree = 4):
        if len(self.X_scaled.shape) == 1:
            self.X_scaled = self.X_scaled.unsqueeze(1)
        # Concatenate a column of ones to has the bias in X
        ones_col = torch.ones((self.X_scaled.shape[0], 1), dtype=torch.float32)
        X_d = torch.cat([ones_col, self.X_scaled], axis=1)
        for i in range(1, degree):
            X_pow = self.X_scaled.pow(i + 1)
            # If we use the gradient descent method, we need to
            # standardize the features to avoid exploding gradients
            if self.standardize_nd_features:
                X_pow -= X_pow.mean()
                std = X_pow.std()
                if std != 0:
                    X_pow /= std
            X_d = torch.cat([X_d, X_pow], axis=1)
        return X_d
    
    def normal_equation(y_true, X):
        """Computes the normal equation

        Args:
            y_true: A torch tensor for the labels.
            X: A torch tensor for the data.
        """
        XTX_inv = (X.T.mm(X)).inverse()
        XTy = X.T.mm(y_true)
        weights = XTX_inv.mm(XTy)
        return weights   
    
    def gradient_descent(X, y_true, lr=0.001, it=30000):
        """Computes the gradient descent

        Args:
            X: A torch tensor for the data.
            y_true: A torch tensor for the labels.
            lr: A scalar for the learning rate.
            it: A scalar for the number of iteration
                or number of gradient descent steps.
        """
        weights_gd = torch.ones((X.shape[1], 1))
        n = X.shape[0]
        fact = 2 / n
        for _ in range(it):
            y_pred = predict(X, weights_gd)
            grad = fact * X.T.mm(y_pred - y_true)
            weights_gd -= lr * grad
        return weights_gd

    def fit(self, features, grad_descent = False):
        y_true = self.y.unsqueeze(1)        
        if grad_descent:
             coef = self.gradient_descent(features, y_true)
        else:
            coef = self.normal_equation(y_true, features)           
        return coef
    
    def predict(features, coef):
        return features.mm(coef).view(1,-1).squeeze().numpy()
    
    