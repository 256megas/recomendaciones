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

print(df.head())
# Cinco primeros datos
print("************")
print(df.info())
# Información de las cabeceras
