import streamlit as st
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


@st.cache
def load_data():
    '''
    Загрузка данных
    '''
    data = load_boston()
    return data


st.sidebar.header('Метод ближайших соседей')
boston = load_data()
neighbors_slider = st.sidebar.slider(
    'Значение гиперпараметра k:', min_value=1, max_value=50, value=10, step=1)

boston_X_train, boston_X_test, boston_y_train, boston_y_test = train_test_split(
    boston.data,
    boston.target,
    test_size=0.2,
    random_state=1
)

neighbors_model = KNeighborsRegressor(n_neighbors=neighbors_slider)
neighbors_model.fit(boston_X_train, boston_y_train)
Y_pred = neighbors_model.predict(boston_X_test)

st.subheader('Оценка качества модели')
st.text('Красные кружки - искомые значения целевого признака')
st.text('Синие кружки - предсказуемые значения целевого признака')

# Изменение качества модели в зависимости от гиперпараметра
dataset_lines = [x for x in range(1, len(boston_X_test) + 1)]

fig1 = plt.figure(figsize=(10, 10))
plt.plot(dataset_lines, boston_y_test, 'ro', dataset_lines, Y_pred, 'bo')
plt.xlabel('Номер записи в датасете')
plt.ylabel('Значение целевого признака')
st.pyplot(fig1)
