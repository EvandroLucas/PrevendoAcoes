import time
import yfinance as yf  
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import sklearn as sk
import numpy as np
from sklearn import svm
from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN, tcn_full_summary
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def add_indicators(stock):
  stock["MovAvgShort"] = None
  stock["MovAvgLong"] = None
  stock["MovAvgDiff"] = None
  K_short = 2 / (10 + 1)
  K_long = 2 / (50 + 1)
  row_iterator = stock.iterrows()
  last_idx, _ = next(row_iterator)
  last_close = stock.at[last_idx,"Close"]
  stock.at[last_idx,"MovAvgShort"] = round( (K_short*last_close) + last_close*(1-K_short) , 4)
  stock.at[last_idx,"MovAvgLong" ] = round( (K_long*last_close ) + last_close*(1-K_long ) , 4)
  stock.at[last_idx,"MovAvgDiff" ] = round( stock.at[last_idx,"MovAvgLong"] - stock.at[last_idx,"MovAvgShort" ] , 4)
  for idx, curr in row_iterator:
    close = stock.at[idx,"Close"]
    stock.at[idx,"MovAvgShort"] = round( (K_short*close) + last_close*(1-K_short) , 4)
    stock.at[idx,"MovAvgLong" ] = round( (K_long*close ) + last_close*(1-K_long ) , 4)
    stock.at[idx,"MovAvgDiff" ] = round( stock.at[idx,"MovAvgShort"] - stock.at[idx,"MovAvgLong" ] , 4)
    last_close = close

  return stock

def add_trends(stock,days = 5):
  colname = "Last"+str(days)+"DaysTrend"
  stock[colname] = None
  last_n = []
  row_iterator = stock.iterrows()
  first_idx, _ = next(row_iterator)
  last_n.append(stock.at[first_idx,"Close"])
  for idx, curr in stock.iterrows():
    close = stock.at[idx,"Close"]
    stock.at[idx,colname] =round( close - last_n[-1] , 4)
    if len(last_n) >= days:
      last_n.pop()
    last_n.insert(0,close)
  return stock

def add_change(stock,days = 5):
  colname = "Last"+str(days)+"DaysChange"
  stock[colname] = None
  last_n = []
  row_iterator = stock.iterrows()
  first_idx, _ = next(row_iterator)
  last_n.append(stock.at[first_idx,"Close"])
  last_idx = first_idx
  for idx, curr in stock.iterrows():
    close = stock.at[idx,"Close"]
    stock.at[last_idx,colname] = round((close - last_n[-1]) / last_n[-1] , 4)
    if len(last_n) >= days:
      last_n.pop()
    last_n.insert(0,close)
    last_idx = idx
  return stock

def add_predictions(stock):
  stock["UpOrDown"] = None

  row_iterator = stock.iterrows()
  last_idx, last = next(row_iterator)
  for idx, curr in row_iterator:
      curr_close = stock.at[idx,"Close"]
      last_close = stock.at[last_idx,"Close"]
      change = 0
      if curr_close > last_close:
        change = 1
      if last_close > curr_close:
        change = -1
      stock.at[last_idx,"UpOrDown"] = change
      last = curr
      last_idx = idx
  stock.at[last_idx,"UpOrDown"] = 0

  return stock

def stock_pre_processing_pipiline(stock,change_days):
  t1 = time.time()
  stock = add_indicators(stock)
  stock = add_trends(stock,days=7)
  stock = add_change(stock,days=change_days)
  stock = add_predictions(stock)
  et = int(time.time() - t1)
  print("Elapsed time: " + str(et) + " seconds")
  # beep(beeps = 2)
  return stock


