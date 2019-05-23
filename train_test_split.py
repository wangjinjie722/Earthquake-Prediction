#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:47:02 2019

@author: ruoqi
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
if __name__ == "__main__":
    path = "./Earthquake/"
    train_df = pd.read_csv(path + "raw.csv")
    X = train_df["time_to_failure"]
    y = train_df["acoustic_data"]
    X_train = X.iloc[:300000000]
    y_train = y.iloc[:300000000]
    X_test = X.iloc[300000001:400000000]
    y_test = y.iloc[300000001:400000000]
    y_ts = pd.DataFrame(data=y_test.values, columns=["acoustic_data"])
    X_ts = pd.DataFrame(data=X_test.values, columns=["time_to_failure"])
    testfile = pd.merge(X_ts, y_ts, left_index=True, right_index=True)
    y_tr = pd.DataFrame(data=y_train.values, columns=["acoustic_data"])
    X_tr = pd.DataFrame(data=X_train.values, columns=["time_to_failure"])
    trainfile = pd.merge(X_tr, y_tr, left_index=True, right_index=True)
    testfile.to_csv(path+"test.csv",index=False)
    trainfile.to_csv(path+"train.csv",index=False)   
