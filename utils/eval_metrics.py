import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, f1_score

def calF1Macro(y_true, y_pred):
    return round(f1_score(y_true, y_pred, average='macro'), 4)

def calF1Micro(y_true, y_pred):
    return round(f1_score(y_true, y_pred, average='micro'), 4)

def calMAPE(y_true, y_pred):
    return round(mean_absolute_percentage_error(y_true, y_pred)*100, 2)

def calMAE(y_true, y_pred):
    return round(mean_absolute_error(y_true, y_pred)/10000, 2)

def calMSE(y_true, y_pred):
    return round(mean_squared_error(y_true, y_pred)/10000, 2)

def calRMSE(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred))/10000, 2)

def calHitRate(y_hat, y, rate):
    hit, total = 0, 0
    for each_y_hat, each_y in zip(y_hat, y):
        if np.abs((each_y_hat-each_y)/each_y) < rate:
            hit += 1
        total += 1
    return round((hit/total)*100, 2)