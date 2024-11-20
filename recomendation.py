# https://365datascience.com/tutorials/how-to-build-recommendation-system-in-python/
# Download Dataset
# https://github.com/XuefengHuang/RecommendationSystem
# C:\Users\#####\AppData\Local\Programs\Python\Python313>
# pip install pandas
# pip install -U scikit-learn
# Panda: Manipulacion y analisis de datos
# scikit-learn: Machine Learning


import pandas as pd

df = pd.read_csv('./recomendationEngine/dataset/BX-Books.csv',
                 encoding='latin-1', sep=';', low_memory=False)
# on_bad_lines omitira las lineas con error
# on_bad_lines='skip',

# Cinco primeros datos
print(df.head())
print("************************************************")
# Información de las cabeceras
print(df.info())
print("************************************************")
# Hay duplicados ? Los borramos
print(df.duplicated(subset='Book-Title').sum())
df = df.drop_duplicates(subset='Book-Title')
print(df.duplicated(subset='Book-Title').sum())
print("************************************************")
# Cojemos una muestra de 15000 libros
sample_size = 15000
# Hacemos una muestra
# replace=false No permite que la muestra se coja de la misma tupla mas de una vez
# random_state es la semilla
df = df.sample(n=sample_size, replace=False, random_state=490)
df = df.reset_index()
df = df.drop('index', axis=1)
print("************************************************")
print(df.head())
