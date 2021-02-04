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
# Web-приложение для решения задачи классификации с учителем
Приложение предназначено для построения моделей классификации, реализованных на языке Python, с использованием библиотек для машинного обучения.
"""
st.header('**1) Загрузите исходные данные**')
st.write('*Данные могут иметь расширение csv или txt (разделители полей - запятые, десятичный разделитель - точка)*')
uploaded_file = st.file_uploader("Выберите файл", ["csv","txt"])

features = list() #признаки
df = pd.DataFrame() # датафрейм для исходных данных
num_of_hidden_layers_MLP = 1

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.subheader('Таблица с исходными данными')
  st.write(df.head())
  features = df.columns

st.header('**2) Укажите признаки, которые будут использоваться для классификации**')
X_features = st.multiselect('Объясняющие признаки', features)
y_feature = st.selectbox('Результативный признак', features)

st.header('**3) Выберите модель для классификации**')
chosen_model = st.selectbox('Модель:', 
                 ('Дерево принятия решений', 'Случайный лес', 
                 'Многослойный персептрон', 'Логистическая регрессия'))

st.header('**4) Настройте параметры модели**')
st.write('*Если оставить поле \"Настраиваемые параметры\" пустым, будут использоваться параметры по умолчанию.*')
is_configurable_params = st.checkbox('Настраиваемые параметры')

if is_configurable_params == True:
  st.sidebar.header('Настройка параметров модели классификации')
  
  if chosen_model == 'Дерево принятия решений':
    st.sidebar.write('Дерево принятия решений')
    criterion_DT = st.sidebar.selectbox('Критерий расщепления:', ('Критерий Джини', 'Энтропия'))
    depth_DT = st.sidebar.slider("Максимальная глубина дерева", min_value=2, max_value=20, value=5, step=1)
    min_samples_split_DT = st.sidebar.slider("Минимальное число объектов для разбиения", min_value=2, max_value=50, value=10, step=1)
    min_samples_leaf_DT = st.sidebar.slider("Минимальное число объектов в листе", min_value=2, max_value=10, value=2, step=1)
  
    params_DT = {'criterion': criterion_DT, 'min_samples_split': min_samples_split_DT, 'min_samples_leaf' : min_samples_leaf_DT, 'max_depth' : depth_DT}
    
  elif chosen_model == 'Случайный лес':
    st.sidebar.write('Случайный лес')
    criterion_RF = st.sidebar.selectbox('Критерий расщепления:', ('Критерий Джини', 'Энтропия'))
    n_estimators_RF = st.sidebar.slider("Число деревьев", min_value=10, max_value=300, value=100, step=5)
    depth_RF = st.sidebar.slider("Максимальная глубина дерева", min_value=2, max_value=15, value=5, step=1)
    min_samples_split_RF = st.sidebar.slider("Минимальное число объектов для разбиения", min_value=2, max_value=50, value=10, step=1)
    min_samples_leaf_RF = st.sidebar.slider("Минимальное число объектов в листе", min_value=2, max_value=10, value=2, step=1)
    max_features_RF = st.sidebar.selectbox('Максимальное число признаков для выбора расщепления:', ('sqrt', 'log2', 'all'))
    
    params_RF = {'criterion': criterion_RF, 'n_estimators': n_estimators_RF, 'max_depth' : depth_RF,
                 'min_samples_split': min_samples_split_RF, 'min_samples_leaf' : min_samples_leaf_RF, 'max_features': max_features_RF}
    
  elif chosen_model == 'Многослойный персептрон':
    st.sidebar.write('Многослойный персептрон')
    num_of_hidden_layers_MLP = st.sidebar.slider("Число скрытых слоев", min_value=1, max_value=5, value=2, step=1)
    hidden_layer_sizes_MLP = st.sidebar.slider("Число нейронов в скрытом слое", min_value=1, max_value=100, value=10, step=1)
    activation_MLP = st.sidebar.selectbox('Функция активации скрытых слоев:', ('Линейная', 'Выпрямленная (ReLU)', 'Гиперболический тангенс', 'Сигмоидальная'))
    solver_MLP = st.sidebar.selectbox('Алгоритм оптимизации параметров:',
                                          ('Стохастический градиентный спуск (SGD)', 'Адаптивная оценка моментов (adam)', 'Квазиньютоновские методы'))
    learning_rate_init_MLP = st.sidebar.number_input('Начальное значение коэффициента обучения', 0.001)
    
    params_MLP = {'hidden_layer_sizes': hidden_layer_sizes_MLP, 'activation': activation_MLP, 'solver' : solver_MLP, 'learning_rate_init': learning_rate_init_MLP}
   
  elif chosen_model == 'Логистическая регрессия':
    st.sidebar.write('Логистическая регрессия')
    penalty_LR = st.sidebar.selectbox('Вид регуляризации:', ('L1', 'L2', 'Без регуляризации'))
    solver_LR = st.sidebar.selectbox('Алгоритм оптимизации параметров:',
                                          ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'))
    max_iter_LR = st.sidebar.slider("Максимальное число итераций", min_value=100, max_value=1000, value=500, step=50)
    
    params_LR = {'penalty': penalty_LR, 'solver': solver_LR, 'max_iter' : max_iter_LR}
  
if st.button('КЛАССИФИЦИРОВАТЬ'):
  st.header('**5) Результаты классификации**') 
  
  #Данные для построения классификатора  
  X = df.loc[:, X_features]
  y = df.loc[:, [y_feature]]
  X = pd.get_dummies(X)
  if (y.dtypes).all() == 'object':
    y = pd.DataFrame({'y' : pd.factorize(y.iloc[:,0])[0]})
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
    
  #-----------------------------------------1 НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ------------------------------------------------
  if is_configurable_params == True:
    #---------------------------------------1.1 ДЕРЕВО ПРИНЯТИЯ РЕШЕНИЙ-----------------------------------------------
    if chosen_model == 'Дерево принятия решений':
      if params_DT['criterion'] == 'Критерий Джини':  params_DT['criterion'] = 'gini'
      elif params_DT['criterion'] == 'Энтропия': params_DT['criterion'] = 'entropy'
        
      clf = tree.DecisionTreeClassifier(**params_DT, random_state = 8)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
    #---------------------------------------1.2 СЛУЧАЙНЫЙ ЛЕС-----------------------------------------------  
    elif chosen_model == 'Случайный лес':
      if params_RF['criterion'] == 'Критерий Джини':  params_RF['criterion'] = 'gini'
      elif params_RF['criterion'] == 'Энтропия': params_RF['criterion'] = 'entropy'
        
      if params_RF['max_features'] == 'all':  params_RF['max_features'] = None
        
      clf = ensemble.RandomForestClassifier(**params_RF, random_state = 8)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
    #---------------------------------------1.3 МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН-----------------------------------------------  
    elif chosen_model == 'Многослойный персептрон':
      if params_MLP['activation'] == 'Линейная':  params_MLP['activation'] = 'identity'
      elif params_MLP['activation'] == 'Выпрямленная (ReLU)':  params_MLP['activation'] = 'relu'
      elif params_MLP['activation'] == 'Гиперболический тангенс':  params_MLP['activation'] = 'tanh' 
      elif params_MLP['activation'] == 'Сигмоидальная':  params_MLP['activation'] = 'logistic'
      
      if params_MLP['solver'] == 'Стохастический градиентный спуск (SGD)':  params_MLP['solver'] = 'sgd'
      elif params_MLP['solver'] == 'Адаптивная оценка моментов (adam)':  params_MLP['solver'] = 'adam'
      elif params_MLP['solver'] == 'Квазиньютоновские методы':  params_MLP['solver'] = 'lbfgs'
      
      hidden_layer_sizes = (params_MLP['hidden_layer_sizes'],)
      for i in range(num_of_hidden_layers_MLP - 1):
          hidden_layer_sizes += (params_MLP['hidden_layer_sizes'],)
      params_MLP['hidden_layer_sizes'] = hidden_layer_sizes
      
      clf = MLPClassifier(**params_MLP, random_state = 8)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      
    #---------------------------------------1.4 ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ-----------------------------------------------  
    elif chosen_model == 'Логистическая регрессия':
      if params_LR['penalty'] == 'L1':  params_LR['penalty'] = 'l1'
      elif params_LR['penalty'] == 'L2':  params_LR['penalty'] = 'l2'
      elif params_LR['penalty'] == 'Без регуляризации':  params_LR['penalty'] = 'none'
      
      clf = LogisticRegression(**params_LR, random_state = 8)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      
  #-----------------------------------------2 ПАРАМЕТРЫ ПО УМОЛЧАНИЮ------------------------------------------------    
  elif is_configurable_params == False:
    
    #---------------------------------------2.1 ДЕРЕВО ПРИНЯТИЯ РЕШЕНИЙ-----------------------------------------------
    if chosen_model == 'Дерево принятия решений':
      clf = tree.DecisionTreeClassifier(random_state = 8)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      
    #---------------------------------------2.2 СЛУЧАЙНЫЙ ЛЕС-----------------------------------------------  
    elif chosen_model == 'Случайный лес':
      clf = ensemble.RandomForestClassifier(random_state = 8)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      
    #---------------------------------------2.3 МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН-----------------------------------------------  
    elif chosen_model == 'Многослойный персептрон':
      clf = MLPClassifier(random_state = 8)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      
    #---------------------------------------2.4 ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ-----------------------------------------------  
    elif chosen_model == 'Логистическая регрессия':     
      clf = LogisticRegression(random_state = 8)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
    
  st.subheader('Метрики качества классификации на тестовой выборке')
  metrics = pd.DataFrame({'Accuracy' : round(clf.score(X_test, y_test), 5),
            'Precision' : round(precision_score(y_test, y_pred), 5),
            'Recall' : round(recall_score(y_test, y_pred), 5),
            'F-мера' : round(f1_score(y_test, y_pred), 5)}, index=['Значение'])

  st.table(metrics)
  st.subheader('Матрица ошибок')
  st.write('*Матрица ошибок показывает сколько объектов класса i были распознаны как объекты класса j.\
           На основании матрицы ошибок рассчитываются основные метрики качества классификации.*')
  
  fig_cm = plt.figure(figsize=(6, 4))
  conf_matrix = confusion_matrix(y_test,y_pred)
  sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='twilight', linewidths=.5)
  plt.xlabel('Предсказанные значения классов')
  plt.ylabel('Истинные значения классов')
  st.pyplot(fig_cm)
  
  st.subheader('ROC-кривая')
  st.write('*ROC-кривая – график, показывающий зависимость верно классифицируемых объектов положительного класса от ложно положительно \
           классифицируемых объектов негативного класса. Иначе, это соотношение True Positive Rate (Recall) и False Positive Rate.*')
  plot_roc_curve(clf, X_test, y_test)
  st.pyplot()
