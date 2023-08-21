import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rand
import math
from scipy.stats import ks_2samp

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

def pre_process(data):
    X = data.drop("signal", axis = 1)
    X = (X-X.mean())/X.std()
    X = X.join(data["signal"])
    return X

def ks_test(data,n_i):
    sig_d = data[data["signal"]==1]
    bkg_d = data[data["signal"]==0]
    sig_d = sig_d.drop("signal",axis=1)
    bkg_d = bkg_d.drop("signal",axis=1)
#    d = {"feature":['none'],"ks_value":[0]}
    D = []
    print(D)

    for i in range(len(sig_d.keys())):
        d1 = sig_d[sig_d.keys()[i]]
        d2 = bkg_d[bkg_d.keys()[i]]
        ks = ks_2samp(d1,d2)[0]
        D.append(ks)
        #df2 = pd.DataFrame([sig_d.keys()[i],ks])
        #D.append(df2,ignore_index =True)
    print(D)
    #D.sort_values(by=["ks_value"])
    #best = D.iloc[0:3,0]
    return best



def build_ann(n_hd,n_i,n_n,test_data,train_data,epochs):
    X_test = test_data.drop('signal', axis =1)
    Y_test = test_data['signal']

    X_train = train_data.drop('signal', axis =1)
    Y_train = train_data['signal']

    model = Sequential()
    model.add(Dense(n_n, input_dim=n_i, activation='sigmoid'))
    for i in range(n_hd-1):
        model.add(Dense(n_n, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train,Y_train,epochs=epochs,batch_size =len(Y_train))
    f_out = model.layers[-1].output
    print(f_out[0])
    eval = model.evaluate(X_test,Y_test)
    print(eval)

if __name__ == '__main__':

    n_hd = 2
    n_i = 3
    n_n = 5
    epochs = 200
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')[:-1]
    test_data = test_data.apply(pd.to_numeric)
    best = ['m_bb','m_wbb','m_wwbb']
    
    train_data = train_data[[str(best[0]),str(best[1]),str(best[2]),'signal']]
    test_data = test_data[[str(best[0]),str(best[1]),str(best[2]),'signal']]
    
    build_ann(n_hd,n_i,n_n,test_data,train_data,2000)