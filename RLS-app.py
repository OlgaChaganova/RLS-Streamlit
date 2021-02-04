import streamlit as st
import pandas as pd
import numpy as np

from sklearn import ensemble, tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, plot_roc_curve

import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)
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
df = pd.DataFrame() # датафрейм для исходных данных

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.subheader('Таблица с исходными данными')
  st.write(df.reshape(1, df.shape[1]))

st.write("График временного ряда")
ax = plt.plot(list(range(1,len(df))), df.iloc[:,0])
_ = plt.title("График временного ряда")
plt.legend()
plt.show()
st.pyplot()

st.header('**2) Укажите степень полинома**')
k = st.slider("Степень полинома", min_value=1, max_value=10, value=3, step=1)

