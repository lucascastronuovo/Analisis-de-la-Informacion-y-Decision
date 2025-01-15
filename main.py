import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

import matplotlib.pyplot as plt
import seaborn as sns
import mlxtend.frequent_patterns
import mlxtend.preprocessing
import numpy as np
import warnings

from textblob import TextBlob
from wordcloud import WordCloud

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

#Segmentación de clientes
data = pd.read_csv("./dataset/coffee-shop-sales-revenue.csv", delimiter='|')
data['datetime'] = pd.to_datetime(data['transaction_date'] + ' ' + data['transaction_time'])
data.head()

# Obtengo las ubicaciones de los locales
locales = data.drop_duplicates(subset=['store_location'])['store_location']
print(locales)

# Extraigo las transacciones por locales
datos_por_local = []
for local in locales:
    ventas = data[data['store_location'] == local][['datetime', 'product_type']]
    datos_por_local.append(ventas)

# Extraigo los tipos de productos que se vendieron en cada local
tmpVentas = []
productos = []
for x in range(len(datos_por_local)):
    tmpVentas.append(datos_por_local[x])
    productos.append(pd.DataFrame(datos_por_local[0]).drop_duplicates('product_type')['product_type'].tolist())    
    productos[x].insert(0, 'datetime')

# Convierto los productos en columnas
MBA_data = []
for x in range(len(productos)):
    MBA_data.append(pd.DataFrame([productos[x]]))
    MBA_data[x].columns = productos[x]

# NO EJECUTAR A MENOS QUE SEA NECESARIO
# Construyo la matriz en base a productos comprados por fecha

# Crear una matriz de "one-hot encoding" usando pivot y fillna para evitar el bucle
for x in range(len(tmpVentas)):
    # Crear una columna 'comprado' con valor 1 para indicar que se compró el producto en esa transacción
    tmpVentas[x]['comprado'] = 1

    # Convertir los datos al formato de "one-hot encoding" usando pivot
    MBA_data[x] = tmpVentas[x].pivot_table(
        index='datetime',
        columns='product_type',
        values='comprado',
        fill_value=0
    ).reset_index()

# Ahora MBA_data[x] debería tener un formato de "one-hot encoding" con menos costo computacional

# Agrupamos las columnas que tienen la misma fecha
MBA_data_grp = []
for x in range(len(MBA_data)):
    MBA_data_grp.append(MBA_data[x].groupby('datetime').sum().reset_index())
    MBA_data_grp[x] = MBA_data_grp[x].drop(columns=['datetime'])
    MBA_data_grp[x] = MBA_data_grp[x].map(lambda x: True if x >= 1 else False)

# Realizamos el análisis
frequent_itemsets = []
rules = []
for x in range(len(MBA_data_grp)):
    frequent_itemsets.append(apriori(MBA_data_grp[x], min_support=0.001, use_colnames=True))
    rules.append(association_rules(frequent_itemsets[x], metric="lift"))

# Mostramos los valores obtenidos
for x in range(len(rules)):
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print(f"Localidad: {locales.iloc[x]}")
    rules[x].sort_values(['support', 'confidence', 'lift'], axis = 0, ascending = False).head(10)


# Arbol de decisión

# Cargar el dataset con el delimitador correcto
ruta_dataset = "./dataset/coffee-shop-sales-revenue.csv"
ventas_producto = pd.read_csv(ruta_dataset, delimiter='|')

# Ver las primeras filas del dataset para confirmar que cargó correctamente
display(ventas_producto.head())

# Extraer el mes y el año de la columna 'transaction_date'
ventas_producto['transaction_date'] = pd.to_datetime(ventas_producto['transaction_date'])
ventas_producto['Mes'] = ventas_producto['transaction_date'].dt.month
ventas_producto['Año'] = ventas_producto['transaction_date'].dt.year

# Verificar que se añadieron las nuevas columnas
display(ventas_producto[['transaction_date', 'Mes', 'Año']].head())

# Especificar el product_type y product_category de interés
producto_interes = 'Scone'  # Cambiar al producto que te interese
categoria_interes = 'Bakery'  # Cambiar a la categoría de interés

# Filtrar las ventas de este producto y categoría
ventas_filtradas = ventas_producto[(ventas_producto['product_type'] == producto_interes) &
                                   (ventas_producto['product_category'] == categoria_interes)]

# Resumir las ventas por mes para los primeros 4 meses
ventas_resumidas = ventas_filtradas.groupby('Mes')['transaction_qty'].sum().loc[1:4]
print(f"Ventas de {producto_interes} en los primeros meses:\n{ventas_resumidas}")

# Mostrar las ventas en los primeros 4 meses
for mes, cantidad in ventas_resumidas.items():
    print(f"En el mes {mes} se vendieron {cantidad} unidades de {producto_interes} ({categoria_interes}).")

# Ignorar advertencias temporales de FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Gráfico de barras para ventas por mes
plt.figure(figsize=(10, 6))
sns.barplot(x=ventas_resumidas.index, y=ventas_resumidas.values, palette="Blues", hue=None)
plt.title(f"Ventas de {producto_interes} en los primeros 4 meses")
plt.xlabel("Mes")
plt.ylabel("Cantidad de ventas")
plt.show();

# Gráfico de línea para ventas por mes y año
plt.figure(figsize=(12, 6))
sns.lineplot(data=ventas_filtradas, x="Mes", y="transaction_qty", hue="Año", marker="o", palette="tab10")
plt.title(f"Distribución de ventas de {producto_interes} ({categoria_interes}) por mes y año")
plt.xlabel("Mes")
plt.ylabel("Cantidad de ventas")
plt.legend(title="Año")
plt.show();

# Seleccionar las características para el modelo de árbol de decisión
X_producto = ventas_filtradas[['Mes', 'Año']]  # Mes y Año
y_producto = ventas_filtradas['transaction_qty']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_producto, y_producto, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo_arbol_producto = DecisionTreeRegressor(max_depth=4, min_samples_split=2, random_state=42)
modelo_arbol_producto.fit(X_train, y_train)

# Definir meses futuros para realizar predicciones (meses 5 a 8)
meses_futuros = pd.DataFrame({
    'Mes': [5, 6, 7, 8],  # Meses futuros
    'Año': [2024] * 4  # Años futuros
})

# Realizar predicciones para los meses futuros
ventas_futuras = modelo_arbol_producto.predict(meses_futuros)

# Mostrar las predicciones para los próximos meses
print(f"Predicciones de ventas para {producto_interes} ({categoria_interes}) en los próximos meses de 2024:")
for mes, prediccion in zip(meses_futuros['Mes'], ventas_futuras):
    print(f"Mes {mes}: {prediccion:.2f} unidades")

# Crear un DataFrame para combinar ventas reales y predicciones
meses_hist = list(ventas_resumidas.index)
ventas_hist = list(ventas_resumidas.values)
pred_df = pd.DataFrame({
    "Mes": meses_hist + list(meses_futuros['Mes']),
    "Cantidad de ventas": ventas_hist + list(ventas_futuras),
    "Tipo": ["Real"] * len(ventas_hist) + ["Predicción"] * len(ventas_futuras)
})

# Gráfico de barras con ventas reales y predicciones
plt.figure(figsize=(10, 6))
sns.barplot(data=pred_df, x="Mes", y="Cantidad de ventas", hue="Tipo", palette="Set1")
plt.title(f"Ventas reales y predicciones para {producto_interes} ({categoria_interes})")
plt.xlabel("Mes")
plt.ylabel("Cantidad de ventas")
plt.show();

# Visualizar el árbol de decisión
plt.figure(figsize=(15, 10))
plot_tree(modelo_arbol_producto, filled=True, feature_names=['Mes', 'Año'], class_names=True, rounded=True, fontsize=10)
plt.title(f"Árbol de Decisión para {producto_interes} ({categoria_interes})")
plt.show();

# Filtrar solo columnas numéricas
numeric_cols = ventas_producto.select_dtypes(include=['number'])

# Mapa de calor de correlación
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="YlGnBu")
plt.title("Matriz de correlación entre variables numéricas")
plt.show();






# Text Mining - Análisis de Sentimiento
# Cargar el dataset
data = pd.read_csv("./dataset/coffee-shop-sales-revenue.csv", delimiter='|')

# Calcular polaridad (sentimiento) de cada descripción
data['sentiment'] = data['product_detail'].apply(lambda x: TextBlob(x).sentiment.polarity)
sentiment_counts = data['sentiment'].apply(lambda x: 'Positivo' if x > 0 else ('Negativo' if x < 0 else 'Neutral')).value_counts()

# Graficar
plt.figure(figsize=(15, 4))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightgrey'])
plt.title("Distribución de sentimientos en las descripciones de productos")
plt.show()

# Calcular polaridad de cada descripción
data['polarity'] = data['product_detail'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Histograma de polaridad
plt.figure(figsize=(10, 4))
sns.histplot(data['polarity'], kde=True, bins=20)
plt.title('Distribución de la Polaridad del Sentimiento')
plt.xlabel('Polaridad')
plt.ylabel('Frecuencia')
plt.show()

# Calcular subjetividad de cada descripción
data['subjectivity'] = data['product_detail'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Histograma de subjetividad
plt.figure(figsize=(10, 4))
sns.histplot(data['subjectivity'], kde=True, bins=20, color="purple")
plt.title('Distribución de la Subjetividad del Sentimiento')
plt.xlabel('Subjetividad')
plt.ylabel('Frecuencia')
plt.show()

# Filtrar las descripciones con polaridad positiva
positive_text = ' '.join(data[data['polarity'] > 0]['product_detail'])
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

plt.figure(figsize=(15, 4))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras - Descripciones Positivas')
plt.show()

# Filtrar las descripciones con polaridad negativa
negative_text = ' '.join(data[data['polarity'] < 0]['product_detail'])
wordcloud_negative = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text)

plt.figure(figsize=(15, 4))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras - Descripciones Negativas')
plt.show()


#Clustering K-means

#Importar datos
df = pd.read_csv("./dataset/coffee-shop-sales-revenue.csv", delimiter = '|')

df

# Simplificación  del análisis
    #Convertimos la variable "transaction_time" en "periodo" para separando los horarios en "Mañana", "Tarde" y "Noche"

df_coffe = pd.DataFrame(df)

df_coffe['transaction_time'] = df_coffe['transaction_time'].str.strip().str[:5]

df_coffe['transaction_time'] = pd.to_datetime(df_coffe['transaction_time'], format='%H:%M').dt.time


def categorize_time(time):
    if time >= pd.to_datetime('06:00').time() and time < pd.to_datetime('12:00').time():
        return 'Mañana'
    elif time >= pd.to_datetime('12:00').time() and time < pd.to_datetime('18:00').time():
        return 'Tarde'
    else:
        return 'Noche'

    # Aplicamos la función a la columna transaction_time
df_coffe['periodo'] = df_coffe['transaction_time'].apply(categorize_time)

    # Eliminamos la variable transaction_time
df_coffe = df_coffe.drop(['transaction_time'], axis=1)


    #Convertimos la variable "transaction_date" en "temporada" para separando las estaciones en "Otoño", "Invierno", "Primavera" y "Verano"

df['transaction_date'] = pd.to_datetime(df['transaction_date'])

def get_season(date):
    if isinstance(date, pd.Timestamp):
        month = date.month
        day = date.day
        
        # Ajustar las estaciones según el día 21
        if month == 9 and day >= 21 or month == 10 or month == 11 or (month == 12 and day < 21):
            return 'Otoño'
        elif month == 12 and day >= 21 or month == 1 or month == 2 or (month == 3 and day < 21):
            return 'Invierno'
        elif month == 3 and day >= 21 or month == 4 or month == 5 or (month == 6 and day < 21):
            return 'Primavera'
        elif month == 6 and day >= 21 or month == 7 or month == 8 or (month == 9 and day < 21):
            return 'Verano'  
    return None  

    # Creamos la columna "temporada" aplicando la función a la columna "transaction_date"
df_coffe['temporada'] = df['transaction_date'].apply(get_season)

    # Eliminamos la variable "transaction_date"
df_coffe = df_coffe.drop(['transaction_date'], axis=1)


df_coffe

df_coffe.info()

    #Para realizar el clustering con K-means es necesario que las variables sean numéricas

label_encoders = {}
for column in df_coffe.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df_coffe[column] = label_encoders[column].fit_transform(df_coffe[column])

df_coffe[df_coffe['transaction_id']==149456]
df_coffe = df_coffe.drop(['transaction_id', 'store_id', 'product_id'], axis=1)
df_coffe


#Normalizamos los valores

df_coffe.describe()
df_coffe_norm = df_coffe.copy()
df_coffe_norm = (df_coffe-df_coffe.min())/(df_coffe.max()-df_coffe.min())

df_coffe_norm.describe()

    #Todos los valores mínmos son igual a cero y todos los valores máximos son igual a 1

#Búsqueda de la cantidad óptima de clusters

    #Calcularemos el "Codo de Jambú" para determinar el número óptimo de clusters según que tan similares son los elementos dentro de cada uno

wcas = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, max_iter = 300)
    kmeans.fit(df_coffe_norm) #Aplico K-means al dataset
    wcas.append(kmeans.inertia_)

plt.plot(range(1,11),wcas)
plt.title("Codo de Jambú")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS") #WCSS es un indicador de qué tan similares son los elementos dentro de los clusters
plt.show

def optimise_k_means(data,max_k):
    means = []
    inertias = []
    
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        
        means.append(k)
        inertias.append(kmeans.inertia_)
        
    fig = plt.subplots(figsize=(10,5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Número de Clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()

optimise_k_means(df_coffe_norm,10)

    #WCSS es la suma de las distancias cuadradas de cada punto al centroide de su cluster. Cuanto más bajo sea su valor, los elementos dentro de cada cluster están más cerca del centroide, lo que significa una mayor similitud entre ellos
    #La cantidad óptima de clusters se determina cuando se observa una disminución drástica del WCSS. A partir de cinco clústers, la disminución del WCSS se vuelve menos pronunciada en comparación con las etapas anteriores
    #La cantidad de cluster a utilizar será cinco

#Aplicación del método Clustering K-Means al dataset de Maven Coffe

clustering = KMeans(n_clusters = 5, max_iter = 300) #Crea el modelo
clustering.fit(df_coffe_norm) #Aplica el modelo al dataset

KMeans(algorithm='auto',copy_x=True,init='k-means++',max_iter=300, n_clusters=5,n_init=10,n_jobs=None,precompute_distances='auto',random_state=None,tol=0.0001,verbose=0)

df_coffe_norm['KMeans_Clusters'] = clustering.labels_  # Agregamos la clasificación de cada elemento según el cluster al que pertenece
df_coffe_norm.head()

    #Visualización de los clusters
        #Se aplica el Análisis de Componentes Principales (PCA) para reducir la dimensionalidad y agrupar las características en la visualización

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_coffe = pca.fit_transform(df_coffe_norm)
pca_df_coffe = pd.DataFrame(data = pca_coffe, columns = ['Componente_1', 'Componente_2'])
pca_df_coffe = pd.concat([pca_df_coffe,df_coffe_norm[['KMeans_Clusters']]], axis=1)

pca_df_coffe

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Componente 1', fontsize=15)
ax.set_ylabel('Componente 2', fontsize=15)
ax.set_title('Componentes Principales', fontsize=20)

color_theme = np.array(["blue", "green", "orange","red","purple"])
# Reducimos la opacidad y el tamaño de los puntos
ax.scatter(x=pca_df_coffe.Componente_1, y=pca_df_coffe.Componente_2, 
           c=color_theme[pca_df_coffe.KMeans_Clusters], s=20, alpha=0.5)

plt.show()


#Interpretación de los resultados
# Mapeo de valores numéricos a texto
mapeo_periodo = {
    0: 'Mañana',
    2.0: 'Tarde',
    1.0: 'Noche'
}

mapeo_temporada = {
    0: 'Invierno',
    1: 'Primavera',
    2: 'Verano',
    3: 'Otoño'
}

# Revertir la conversión usando el diccionario
df_coffe['periodo'] = df_coffe['periodo'].map(mapeo_periodo)
df_coffe['temporada'] = df_coffe['temporada'].map(mapeo_temporada)


df['periodo'] = df_coffe['periodo'] #agregamos al DataFrame original las columnas "periodo", "temporada" y "KMeans_Clusters"
df['temporada'] = df_coffe['temporada']
df['KMeans_Clusters'] = df_coffe_norm['KMeans_Clusters'] 

df

cluster_0 = df[df['KMeans_Clusters'] == 0]
cluster_1 = df[df['KMeans_Clusters'] == 1]
cluster_2 = df[df['KMeans_Clusters'] == 2]
cluster_3 = df[df['KMeans_Clusters'] == 3]
cluster_4 = df[df['KMeans_Clusters'] == 4]

    #Cluster 0
cluster_0

    #Cluster 1
cluster_1

    #Cluster 2
cluster_2

    #Cluster 3
cluster_3

    #Cluster 4
cluster_4


    #Resumen

cluster_summary = df.groupby('KMeans_Clusters').agg({
    'store_location': lambda x: x.value_counts().index[0],  # Ubicación más común
    'periodo': lambda x: x.value_counts().index[0],         # Hora más común
    'temporada': lambda x: x.value_counts().index[0],# Temporada más común
    'product_category': lambda x: x.value_counts().index[0],# Categoría más común
    'transaction_id': 'count'                               # Cantidad de transacciones
})

print(cluster_summary)

    #Gráfico de barras de transacciones por localidad de cada cluster
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='KMeans_Clusters', hue='store_location')
plt.title("Distribución de ubicaciones por cluster")
plt.show()

    #Gráfico de barras de transacciones por periodo de cada cluster
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='KMeans_Clusters', hue='periodo')
plt.title("Distribución de periodos por cluster")
plt.show()


    #Gráfico de barras de transacciones por temporada de cada cluster
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='KMeans_Clusters', hue='temporada')
plt.title("Distribución de temporadas por cluster")
plt.show()


    #Gráfico de barras de transacciones por categoría de cada cluster
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='KMeans_Clusters', hue='product_category')
plt.title("Distribución de categorías por cluster")
plt.show()


    #Insights

        #Grupo A
            #- Suelen concurrir a las tres localidades, pero sobre todo a Lower Manhattan
            #- Suelen ir a la mañana y poco a la noche
            #- Tienen concurrencia todo el año, pero mayormente en primavera
            #- Consumen mucho té y un poco de chocolate


        #Grupo B
            #- Concurren principalmente a la localidad de Lower Manhattan, pero también a Hell’s Kitchen. No van a Astoria
            #- Suelen ir principalmente a la mañana y un poco a la noche
            #- Tienen concurrencia todo el año, pero mayormente en primavera
            #- Consumen mucho café y también bastante bakery, chocolatada y flavours

        #Grupo C
            #- Suelen concurrir a las tres localidades, pero sobre todo a Hell’s Kitchen
            #- Suelen ir a la tarde y cada tanto a la noche. No van a la mañana
            #- Tienen concurrencia todo el año, pero mayormente en primavera
            #- Consumen mucho café, bastante bakery y chocolatada

        #Grupo D
            #- Concurren principalmente a la localidad de Astoria, cada tanto van a Hell’s Kitchen. No van a Lower Manhattan
            #- Suelen ir principalmente a la mañana y cada tanto a la noche. No van a la tarde
            #- Tienen concurrencia todo el año, pero mayormente en primavera
            #- Consumen mucho café, bastante bakery y cada tanto chocolatada

        #Grupo E
            #- Suelen concurrir a las tres localidades, pero sobre todo a Astoria
            #- Suelen ir principalmente a la tarde y cada tanto a la noche
            #- Tienen concurrencia todo el año, pero mayormente en primavera
            #- Consumen mucho té y un poco de chocolate


    #Observaciones Generales
        #- El grupo B no va a Astoria y el grupo D no va a Lower Manhattan. El resto suele ir a las tres localidades
        #- Los grupos A, B y D suelen ir mucho a la mañana, cada tanto a la noche y no van a la tarde. En cambio, los grupos C y E van mucho a la tarde, cada tanto a la noche y no van a la mañana
        #- En todos los grupos se tiene más concurrencia en primavera, le sigue invierno y luego verano
        #- A nivel general, se suele consumir café, té, bakery, chocolatada y flavours
        

