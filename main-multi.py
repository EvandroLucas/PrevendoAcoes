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
from aux import *
from pandas.plotting import register_matplotlib_converters
from datetime import datetime
import json
import os
import time
register_matplotlib_converters()

def beep(times = 1):
  for i in range(times):
    duration = 0.5  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    time.sleep(duration)

def gain(x,y):
  if x > y:
    return 1 - (y/x)
  else:
    return 1 - (x/y)

def loss(x,y):
  if x > y:
    return round((x/y) -1,4)
  else:
    return round((y/x) -1,4)

def movAvgExp(series,cycle):
  movavg = series.copy()
  last = series[0]
  K = 0.7 / (cycle + 1)
  for i in range(1,len(series)):
    movavg[i] = (K*series[i])  + last*(1-K)
    last = movavg[i]
  return movavg


def print_results(stk,decision,stock_code):
  plt.figure(figsize=(16,8))
  startDate = '2017-01-01'
  endDate = '2022-01-01'
  stk_tmp = stk[(stk.index > startDate) & (stk.index <= endDate)]
  stk_tmp['Close'].plot(label="Close")
  plt.xlabel("Date")
  plt.ylabel("Close")
  plt.title(stock_code + " Price data")
  for index, d in decision.iterrows():
    if d['Value'] == 0:
      plt.axvline(pd.Timestamp(index),color='red')
    if d['Value'] == 1:
      plt.axvline(pd.Timestamp(index),color='green')
  plt.show()

def check_decisions(stk,decision,stockCode):

  results = {
    'dataInicio' : str(datetime.strptime(str(stk.index[0]), '%Y-%m-%d %H:%M:%S')) ,
    'dataFim' : str(datetime.strptime(str(stk.index[-1]), '%Y-%m-%d %H:%M:%S')),
    'stockCode' : stockCode,
    'numeroPontos' : 1,
    'acertos' : 0,
    'erros' : 0,
    'acertosPct' : 0,
    'errosPct' : 0,
    'ganhoRel' : 0,
    'ganhoMin' : 0,
    'ganhoMed' : 0,
    'ganhoMax' : 0,
    'perdaRel' : 0,
    'perdaMin' : 0,
    'perdaMed' : 0,
    'perdaMax' : 0
  }

  ganhos = []
  perdas = []

  row_iterator = decision.iterrows()
  last_index = next(row_iterator)[0]
  last_price = stk.at[last_index,'Close']
  last_value = 0.5
  for index, d in decision.iterrows():
    decision.at[index,'Price'] = stk.at[index,'Close']
    curr_price = decision.at[index,'Price']
    if d['Value'] != 0.5:
      results['numeroPontos'] += 1
      if last_value == 0.5:
        if d['Value'] == 1:
          last_value = 0
        if d['Value'] == 0:
          last_value = 1
      if last_value == 1 : 
        if curr_price > last_price:
          decision.at[last_index,'Guess'] = 'Right'
          results['acertos'] += 1
          ganhos.append(gain(curr_price,last_price))
        else:
          decision.at[last_index,'Guess'] = 'Wrong'
          results['erros'] += 1
          perdas.append(loss(curr_price,last_price))
      if last_value == 0 : 
        if curr_price > last_price:
          decision.at[last_index,'Guess'] = 'Wrong'
          results['erros'] += 1
          perdas.append(loss(curr_price,last_price))
        else:
          decision.at[last_index,'Guess'] = 'Right'
          ganhos.append(gain(curr_price,last_price))
          results['acertos'] += 1
      last_index = index
      last_price = curr_price
      last_value = d['Value']

  final_price = stk['Close'].iloc[-1]
  if last_price < final_price:
    if last_value == 1 : 
      decision.at[last_index,'Guess'] = 'Right'
      ganhos.append(gain(final_price,last_price))
      results['acertos'] += 1
    else : 
      decision.at[last_index,'Guess'] = 'Wrong'
      results['erros'] += 1
      perdas.append(loss(final_price,last_price))
  else:
    if last_value == 1 : 
      decision.at[last_index,'Guess'] = 'Wrong'
      results['erros'] += 1
      perdas.append(loss(final_price,last_price))
    else : 
      decision.at[last_index,'Guess'] = 'Right'
      ganhos.append(gain(final_price,last_price))
      results['acertos'] += 1

  for index, d in decision.iterrows():
    if d['Value'] != 0.5:
      print(index, decision.loc[index]['Value'], d['Guess'],round(d['Price'],2))
  print(stk.index[-1],"---","-----",round(final_price,2))

  results['acertosPct'] = round(results['acertos'] / results['numeroPontos'],4)
  results['errosPct'] = round(results['erros'] / results['numeroPontos'],4)

  print("Ganhos: ", [round(num, 2) for num in ganhos])
  print("Perdas: ", [round(num, 2) for num in perdas])

  results['ganhoMax'] = round(max(ganhos,default=0),4)
  results['ganhoMin'] = round(min(ganhos,default=0),4)
  if len(ganhos) > 0:
    results['ganhoMed'] = round(sum(ganhos) / len(ganhos),4)
  results['ganhoRel'] = round(results['ganhoMed'] * results['acertosPct'],4)

  results['perdaMax'] = round(max(perdas,default=0),4)
  results['perdaMin'] = round(min(perdas,default=0),4)
  if len(perdas) > 0:
    results['perdaMed'] = round(sum(perdas) / len(perdas),4)
  results['perdaRel'] = round(results['perdaMed'] * results['errosPct'],4)

  json_object = json.dumps(results, indent = 2) 
  print(json_object)
  
  with open(output_file, "a") as text_file:
    if os.stat(output_file).st_size == 0:
      for v in results:
        text_file.write(str(v) + ',')
      text_file.write('\n')
    for v in results.values():
      text_file.write(str(v) + ',')
    text_file.write('\n')

  return decision


def train_stock(stk,stock_code,model):
  print("Treinando : " + str(stock_code))
  try:
    stk = stock_pre_processing_pipiline(stk,change_days)
  except StopIteration:
    raise StopIteration("Error: invalid interval for ",stock_code)
    return

  df = stk
  print("Criando X e Y")

  X = df.copy()
  del X['Last20DaysChange']
  del X['UpOrDown']
  y = df.copy()[['Last20DaysChange']]

  X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)
  X_train = np.asarray(X_train).astype('float32')
  y_train = np.asarray(y_train).astype('float32')

  scaler = MinMaxScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  model.fit(X_train_scaled, y_train)

def predict_stock(stk,stock_code,model):

  try:
    stk = stock_pre_processing_pipiline(stk,change_days)
  except StopIteration:
    raise StopIteration("Error: invalid interval for ",stock_code)
    return
  df = stk
  original_columns = df.columns
  dataset = df.values

  print("Criando X e Y")

  X = df.copy()
  del X['Last20DaysChange']
  del X['UpOrDown']
  y = df.copy()[['Last20DaysChange']]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)
  X_train = np.asarray(X_train).astype('float32')
  X_test = np.asarray(X_test).astype('float32')
  y_train = np.asarray(y_train).astype('float32')
  y_test = np.asarray(y_test).astype('float32')

  scaler = MinMaxScaler()
  X_train_scaled = scaler.fit_transform(X_train)

  X_test_scaled = scaler.transform(X_test)
  y_pred = model.predict(X_test_scaled)

  y_test2 = np.concatenate( y_test, axis=0 )
  y_test3 = sm.nonparametric.lowess(y_test2, np.arange(0, y_test2.size), frac = 0.05)[:,1]  
  #y_test3 = movAvgExp(y_test2,1)
  y_test4 = (y_test3 -  y_test3.min()) / (y_test3.max() - y_test3.min())
  y_test5 = pd.DataFrame(y_test4,columns=['Value'])
  y_test5['Date'] = df.index[-1*(y_test5.size):]
  y_test5 = y_test5.set_index('Date')

  y_pred2 = np.concatenate( y_pred, axis=0 )
  y_pred3 = movAvgExp(y_pred2,1)
  y_pred3 = sm.nonparametric.lowess(y_pred3, np.arange(0, y_pred3.size), frac = 0.05)[:,1]
  y_pred4 = (y_pred3 -  y_pred3.min()) / (y_pred3.max() - y_pred3.min())
  y_pred5 = pd.DataFrame(y_pred4,columns=['Value'])
  y_pred5['Date'] = df.index[-1*(y_pred5.size):]
  y_pred5 = y_pred5.set_index('Date')


# Getting decision based on the first-degree change

  pred_inflection_points = []
  last_p = y_pred5['Value'][0]
  for p in y_pred5['Value']:
    if p > 0.5 and last_p <= 0.5:
      dec_value = 1
    elif p < 0.5 and last_p >= 0.5:
      dec_value = 0
    else:
      dec_value = 0.5
    pred_inflection_points.append((dec_value,'None',0.0))
    last_p = p

  decision = pd.DataFrame(pred_inflection_points,columns=['Value','Guess','Price'])
  decision['Date'] = df.index[-1*(decision.shape[0]):]
  decision = decision.set_index('Date')

  # Checking if decision is right for tendency change

  decision = check_decisions(stk,decision,stock_code)

  # print_results(stk,decision)
  # decision = decision.loc[decision['Value'] != 0.5]
  return decision


########################################################################
########################################################################


stock_codes = [
  'AAPL',
  'TSLA',
  'WMT',
  'KO',
  'AMZN',
  'F',
  'SAN',
  'PHG',
  'MSFT',
  'JPM',
  'DIS',
  'PFE',
  'XOM',
  'T',
  'MCD',
  'BA',
  'GE',
  'NFLX',
  'INTC',
  'GOOG',
  'BAC',
  'BABA',
  'ABEV',
  'PBR',
  'MTCH',
  'AAL',
  'TME',
  'MRNA',
  'ITUB',
  'NOK',
  'ATVI',
  'UBER',
  'VZ',
  'BBD',
  'HPE',
  'X',
  'GM',
  'FB',
  'ABNB',
  'ORCL'
]
repeat = 3
years = 21
intervals = []
for i in range(0,(22-years)):
    intervals.append([str(2000 + i)+'-01-01',str(2000 + i + years)+'-01-01'])

output_file = 'output'+str(years)+'x'+str(years)+'-multi.csv'

for i in range(repeat):
  for itv in intervals:
    model = Sequential()
    model.add(Dense(5000))
    model.add(Dense(1000))
    model.add(Dense(120))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    stocks = {}
    for stock_code in stock_codes:
      stocks[stock_code] = yf.download(stock_code,itv[0],itv[1])

    for stock_code in stock_codes:
      change_days = 20
      try:
        s = stocks[stock_code]
        train_stock(s,stock_code,model)
      except StopIteration:
        pass
        print("Skipping: Data unavailable for " + stock_code + " at ",itv[0],itv[1])
        continue

    for stock_code in stock_codes:
      change_days = 20
      try:
        s = stocks[stock_code]
        predict_stock(s,stock_code,model)
      except:
        pass
        print("Skipping!!!: Data unavailable for " + stock_code + " at ",itv[0],itv[1])
        beep(1)
        continue

try:
  beep(5)
except:
  pass
  print("Can't beep. Please run 'sudo apt install sox' if you want the code to beep at the end.")



