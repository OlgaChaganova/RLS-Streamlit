import streamlit as st
import pandas as pd
import numpy as np
import math 

import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)



class RLS:
    def __init__(self, num_vars, lam, delta):
        '''
        num_vars: number of variables including constant
        lam: forgetting factor, usually very close to 1.
        '''
        self.num_vars = num_vars
        
        # delta controls the initial state.
        self.A = delta*np.matrix(np.identity(self.num_vars))
        self.w = np.matrix(np.zeros(self.num_vars))
        self.w = self.w.reshape(self.w.shape[1],1)
        
        # Variables needed for add_obs
        self.lam_inv = lam**(-1)
        self.sqrt_lam_inv = math.sqrt(self.lam_inv)
        
        # A priori error
        self.a_priori_error = 0
        
        # Count of number of observations added
        self.num_obs = 0

    def add_obs(self, x, t):
        '''
        Add the observation x with label t.
        x is a column vector as a numpy matrix
        t is a real scalar
        '''            
        z = self.lam_inv*self.A*x
        alpha = float((1 + x.T*z)**(-1))
        self.a_priori_error = float(t - self.w.T*x)
        self.w = self.w + (t-alpha*float(x.T*(self.w+t*z)))*z
        self.A -= alpha*z*z.T
        self.num_obs += 1
        
    def fit(self, X, y):
        '''
        Fit a model to X,y.
        X and y are numpy arrays.
        Individual observations in X should have a prepended 1 for constant coefficient.
        '''
        for i in range(len(X)):
            x = np.transpose(np.matrix(X[i]))
            self.add_obs(x,y[i])


    def get_error(self):
        '''
        Finds the a priori (instantaneous) error. 
        Does not calculate the cumulative effect
        of round-off errors.
        '''
        return self.a_priori_error
    
    def predict(self, x):
        '''
        Predict the value of observation x. x should be a numpy matrix (col vector)
        '''
        return float(self.w.T*x)

"""
# Прогнозирование временных рядов с помощью рекуррентного метода наименьших квадратов
Рекуррентный метод наименьших квадратов – это итеративная процедура оценки параметров регрессионной модели, позволяющая уточнять оценки по мере получения новых данных.
Рекуррентный МНК применим в условиях, когда необходимо принимать решение на основе информации, поступающей в реальном времени.
Также РМНК является одним из методов оценки параметров регрессионной модели в случае мультиколлинеарности объясняющих признаков. 
"""
st.header('**1) Загрузите исходный временной ряд**')
st.write('*Данные могут иметь расширение csv или txt (разделители полей - запятые, десятичный разделитель - точка)*')
uploaded_file = st.file_uploader("Выберите файл", ["csv","txt"])

features = list() #признаки
y = pd.DataFrame() # датафрейм для исходных данных

if uploaded_file is not None:
  y = pd.read_csv(uploaded_file, header=None, decimal=',', sep=' ')
  st.subheader('Таблица с исходными данными')
  st.dataframe(np.array(y).reshape(1, len(y)))
  
  st.subheader('График временного ряда')
  ax = plt.plot(list(range(len(y))), y.iloc[:,0])
  st.pyplot()

y = np.array(y)
test_size = len(y)

st.header('**2) Задайте параметры модели**')
k = st.slider("Степень полинома", min_value=1, max_value=10, value=3, step=1)
lam = st.number_input('Начальное значение коэффициента обучения', min_value=0.0, max_value=1.0, value=0.9)
delta = st.number_input('Дельта', min_value=0, max_value=100, value=10)

LS = RLS(k, lam, delta)
pred_x = []
pred_y = []
pred_error = []
for i in range(test_size):
    x = np.matrix(np.zeros((1,k)))
    for j in range(num_vars):
        x[0,j] = i**j 
    pred_x.append(i)
    pred_y.append(float(x*LS.w))
    pred_error.append(LS.get_error())
    LS.add_obs(x.T,y[i])
ax = plt.plot(pred_x[50:], pred_y[50:],label='predicted')
_ = plt.plot(pred_x[50:],y[50:],label='actual')
_ = plt.title("SPY closing price, 8/30/16 - 7/11/18")
plt.legend()
st.pyplot()
