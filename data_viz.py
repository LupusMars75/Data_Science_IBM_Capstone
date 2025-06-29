# Predicción del aterrizaje de la primera etapa del Falcon 9 de SpaceX
# 
# Asignación: Exploración y preparación de datos
# Tiempo estimado necesario: 70 minutos
# En esta tarea, predeciremos si la primera etapa del Falcon 9 aterrizará con éxito.
#  SpaceX anuncia lanzamientos de cohetes Falcon 9 en su sitio web con un coste de 62 millones de dólares;
#  otros proveedores cuestan más de 165 millones de dólares cada uno, gran parte del ahorro se debe a que SpaceX
#  puede reutilizar la primera etapa.
# 
# En este laboratorio, realizará Análisis Exploratorio de Datos e Ingeniería de Características.
# La primera etapa del Falcon 9 aterrizará con éxito

# pandas es una biblioteca de software escrita para el lenguaje de programación Python para la manipulación y análisis de datos.
import pandas as pd
#NumPy es una librería para el lenguaje de programación Python, que añade soporte para matrices y arrays multidimensionales de gran tamaño, junto con una gran colección de funciones matemáticas de alto nivel para operar sobre estos arrays
import numpy as np
# Matplotlib es una librería de ploteo para python y pyplot nos proporciona un marco de ploteo similar a MatLab. Lo utilizaremos en nuestra función de trazado para trazar los datos.
import matplotlib.pyplot as plt
#Seaborn es una librería de visualización de datos de Python basada en matplotlib. Proporciona una interfaz de alto nivel para dibujar gráficos estadísticos atractivos e informativos
import seaborn as sns


#Análisis exploratorio de datos
# En primer lugar, leamos el conjunto de datos de SpaceX en un marco de datos de Pandas e imprimamos su resumen
import requests
import io
import pandas as pd

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
response = requests.get(URL)
dataset_part_2_csv = io.BytesIO(response.content)

df = pd.read_csv(dataset_part_2_csv)
print(df.head(5))

# En primer lugar, intentemos ver cómo las variables FlightNumber (que indica los continuos intentos de lanzamiento)
#  y Payload afectan al resultado del lanzamiento.
# Podemos graficar el FlightNumber vs. PayloadMassy superponer el resultado del lanzamiento. 
# Vemos que a medida que aumenta el número de vuelos, es más probable que la primera etapa aterrice con éxito.
#  La masa de la carga útil también parece ser un factor; incluso con cargas más masivas, la primera etapa a menudo regresa con éxito.
sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()
# A continuación, vamos a desglosar cada sitio para visualizar sus registros de lanzamiento detallados.


#TAREA 1: Visualizar la relación entre el Número de Vuelo y el Sitio de Lanzamiento
#Use la función catplot para graficar Número de Vuelo vs Sitio de Lanzamiento, establezca el parámetro x como Número de Vuelo,
#  establezca y como Sitio de Lanzamiento y establezca el parámetro hue como 'class'.

# Traza un gráfico de puntos de dispersión con el eje x como número de vuelo y el eje y como lugar de lanzamiento,
#  y el matiz como el valor de la clase.
sns.catplot(y="LaunchSite", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20) 
plt.ylabel("Launch Site",fontsize=20)
plt.show()

#TAREA 2: Visualizar la relación entre la masa de la carga útil y el lugar de lanzamiento
# También queremos observar si existe alguna relación entre los lugares de lanzamiento y su masa de carga útil.

# Trazar un gráfico de puntos de dispersión con el eje x para ser la masa de carga útil (kg)
#  y el eje y para ser el lugar de lanzamiento, y el matiz para ser el valor de la clase.
sns.catplot(y="LaunchSite", x="PayloadMass", hue="Class", data=df, aspect = 5)
plt.xlabel("Payload Mass (kg)",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()
# Ahora bien, si se observa el gráfico de puntos de dispersión de la masa de carga útil en función del lugar de lanzamiento, 
# se comprobará que en el lugar de lanzamiento de VAFB-SLC no se lanzan cohetes con una masa de carga útil elevada (superior a 10000).

# TAREA 3: Visualizar la relación entre la tasa de éxito de cada tipo de órbita
# A continuación, queremos comprobar visualmente si existe alguna relación entre la tasa de éxito y el tipo de órbita.
# Vamos a crear un gráfico de barras para la tasa de éxito de cada órbita
# SUGERENCIA use el método groupby en la columna Orbit y obtenga la media de la columna Class   
# Agrupar por órbita y calcular la media de la clase para obtener la tasa de éxito
orbit_success_rate = df.groupby('Orbit')['Class'].mean().reset_index()
# Trazar un gráfico de barras para la tasa de éxito de cada órbita
sns.barplot(x='Orbit', y='Class', data=orbit_success_rate)
plt.xlabel("Orbit", fontsize=20)
plt.ylabel("Success Rate", fontsize=20)
plt.title("Success Rate by Orbit", fontsize=20)
plt.xticks(rotation=45)
plt.show()
#Analiza el gráfico de barras para identificar las órbitas con mayor porcentaje de éxito.


#TAREA 4: Visualizar la relación entre FlightNumber y Orbit type
# 
# Para cada órbita, queremos ver si hay alguna relación entre FlightNumber y Orbit type.
# Traza un gráfico de puntos de dispersión con eje x para ser FlightNumber y eje y para ser la Órbita, y hue para ser el valor de la clase
sns.catplot(y="Orbit", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number", fontsize=20)
plt.ylabel("Orbit", fontsize=20)
plt.show()
# # Se puede observar que en la órbita LEO, el éxito parece estar relacionado con el número de vuelos.
#  Por el contrario, en la órbita GTO, no parece haber relación entre el número de vuelos y el éxito.

# TAREA 5: Visualizar la relación entre la masa de la carga útil y el tipo de órbita
# Del mismo modo, podemos trazar gráficos de puntos de dispersión de la masa de la carga útil
#  frente a la órbita para revelar la relación entre la masa de la carga útil y el tipo de órbita.
# Trazar un gráfico de puntos de dispersión con el eje x para ser la masa de la carga útil y el eje y para ser la órbita, 
# y el matiz para ser el valor de la clase
sns.catplot(y="Orbit", x="PayloadMass", hue="Class", data=df, aspect = 5)
plt.xlabel("Payload Mass (kg)", fontsize=20)    
plt.ylabel("Orbit", fontsize=20)
plt.show()
# 
# Con cargas pesadas, la tasa de aterrizajes con éxito o positivos es mayor para Polar,LEO e ISS.
# Sin embargo, en el caso de GTO, es difícil distinguir entre aterrizajes con éxito y sin éxito, ya que se dan ambos resultados.

#TAREA 6: Visualizar la tendencia anual del éxito del lanzamiento
# 
# Puede trazar un gráfico de líneas con el eje x como año y el eje y como tasa media de éxito,
#  para obtener la tendencia media del éxito del lanzamiento.
# La función le ayudará a obtener el año a partir de la fecha:
# Una función para Extraer años de la fecha
year=[]
def Extract_year():
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year
Extract_year()
df['Date'] = year
df.head()
# Trazar un gráfico de líneas con el eje x para el año extraído y el eje y para la tasa de éxito.
sns.lineplot(x='Date', y='Class', data=df, ci=None)
plt.xlabel("Year", fontsize=20)
plt.ylabel("Success Rate", fontsize=20)
plt.title("Success Rate by Year", fontsize=20)
plt.xticks(rotation=45)
plt.show()

#Features Engineering
# Por ahora, usted debe obtener algunas ideas preliminares acerca de cómo cada variable importante afectaría a la tasa de éxito, 
# vamos a seleccionar las características que se utilizarán en la predicción de éxito en el módulo futuro.
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
print(features.head())




#TAREA 7: Crear variables ficticias para columnas categóricas
# Utilice la función get_dummies y features dataframe para aplicar OneHotEncoder a la columna Orbits,
#  LaunchSite, LandingPad, y Serial. Asigne el valor a la variable features_one_hot, muestre los resultados utilizando el método head. 
# Su dataframe de resultados debe incluir todas las características incluyendo las codificadas.
features_one_hot = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'], drop_first=True)
print(features_one_hot.head())



#TAREA 8: Convertir todas las columnas numéricas a float64
# 
# Ahora que nuestro marco de datos features_one_hot sólo contiene números,
#  transforme todo el marco de datos en una variable de tipo float64
# HINT: use astype function
features_one_hot = features_one_hot.astype('float64')
print(features_one_hot.dtypes)

features_one_hot.to_csv('dataset_part_3.csv', index=False)

# Pregunta 2
# ¿Cuál es el número total de columnas en el marco de datos de características después de aplicar 
# una codificación en caliente a las columnas Orbits, LaunchSite, LandingPad y Serial .
# Aquí el marco de datos de características consta de las siguientes columnas FlightNumber', 'PayloadMass', 'Orbit',
#  'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial'
total_columns = features_one_hot.shape[1]
print(f"Total number of columns in the features dataframe after one-hot encoding: {total_columns}")