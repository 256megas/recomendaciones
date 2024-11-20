# https://365datascience.com/tutorials/how-to-build-recommendation-system-in-python/
# Download Dataset
# https://github.com/XuefengHuang/RecommendationSystem
# C:\Users\#####\AppData\Local\Programs\Python\Python313>
# pip install pandas
# pip install -U scikit-learn
# Panda: Manipulacion y analisis de datos
# scikit-learn: Machine Learning


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('./1_recomendation/dataset/BX-Books.csv',
                 encoding='latin-1', sep=';', low_memory=False)
# on_bad_lines omitira las lineas con error
# on_bad_lines='skip',

# Cinco primeros datos
# print(df.head())
# Información de las cabeceras
# print(df.info())
# Hay duplicados ? Los borramos
# print(df.duplicated(subset='Book-Title').sum())
df = df.drop_duplicates(subset='Book-Title')
# print(df.duplicated(subset='Book-Title').sum())
# Cojemos una muestra de 15000 libros
sample_size = 15000
# Hacemos una muestra
# replace=false No permite que la muestra se coja de la misma tupla mas de una vez
# random_state es la semilla
df = df.sample(n=sample_size, replace=False, random_state=490)
df = df.reset_index()
df = df.drop('index', axis=1)
# print(df.info())
# print(df.head())
print("************************************************")
# print(df.head())
# Adaptamos el contenido para que no haya confusiones


def clean_text(author):
    result = str(author).lower()
    return (result.replace(' ', ''))


df['Book-Author'] = df['Book-Author'].apply(clean_text)
df['Book-Title'] = df['Book-Title'].str.lower()
df['Publisher'] = df['Publisher'].str.lower()
# Quitamos las columnas que no nos interesan:
df2 = df.drop(['ISBN', 'Image-URL-S', 'Image-URL-M',
              'Image-URL-L', 'Year-Of-Publication'], axis=1)
df2['data'] = df2[df2.columns[1:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)

# print(df2.head())
# print(df2['data'].head())
# Vectorizamos
# Step 4: Building the Recommendation System
