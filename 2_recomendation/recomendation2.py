# https://drlee.io/building-a-content-based-recommender-system-with-python-and-google-colab-c753c9bdd449# C:\Users\#####\AppData\Local\Programs\Python\Python313>
# pip install pandas
# pip install -U scikit-learn
# https://raw.githubusercontent.com/fenago/datasets/refs/heads/main/sample-data.csv# Panda: Manipulacion y analisis de datos
# scikit-learn: Machine Learning
# https://www.youtube.com/watch?v=e9U0QAFbfLI

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
# El tf idf vectorizer python tiene en cuenta el número de veces
# que aparece la palabra (o token) en dicho documento, pero también
#  el total de veces que aparece en todo el corpus.
# https://keepcoding.io/blog/que-es-el-algoritmo-tf-idf-vectorizer/
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('./2_recomendation/dataset/clothes.csv')
# Analizamos por palabra
# ngram numero de simbolos adyacentes
# min-df elimina palabras que aparezcan menos de 1%
# stop_words palabras vacias
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3),
                     min_df=0.0, stop_words='english')
tfidf_matrix = tf.fit_transform(data['description'])
# Calcula el núcleo lineal entre X y Y.
# Que vectores de X se convierten en el vector cero de W
# Vector nulo es ortogonal a cualquier orto vector en su espacio
# El núcleo de una matriz, también llamado espacio nulo, es el núcleo
#  de la aplicación lineal definida por la matriz.
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
results = {}
# itera sobre columnas como pares
for idx, row in data.iterrows():
    # argsort puntua elementos por similaridad
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], data['id'][i])
                     for i in similar_indices]

    results[row['id']] = similar_items[1:]

# recuperamos la descripción basandonos en el id del elemento


def item(id):
    return data.loc[data['id'] == id]['description'].tolist()[0].split(' - ')[0]


def recommend(item_id, num):
    print("Recomiendo " + str(num) +
          " productos similares a: " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recomendado: " + item(rec[1]) +
              " (Puntuacion:" + str(rec[0]) + ")")


recommend(item_id=181, num=5)
