from sklearn import preprocessing
import numpy as np


class PolyRegNp():
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        
    def model_compute(X, coef):
        degree = len(coef)-1
        curve = [np.sum([coef[-1]] + [x**(degree-d)*c for d,c \
                in enumerate(coef[:-1])]) for x in X]
        return curve

    def feature_scaling(self):
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.X,self.y)
        X_scaled = scaler.transform(self.X)
        return X_scaled
    
    def fit(self, X, y, degree = 4):
        coef = np.polyfit(X.flatten(), y, degree)
        y_pred = model_compute(X, coef)
        return coef, y_pred