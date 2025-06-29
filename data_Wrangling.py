# En este laboratorio, realizaremos algunos Análisis Exploratorios de Datos (AED) para encontrar algunos patrones en los datos y
#  determinar cuál sería la etiqueta para entrenar modelos supervisados.

#En el conjunto de datos, hay varios casos diferentes en los que el cohete no aterrizó con éxito. Por ejemplo,
#  Océano Verdadero significa que la misión aterrizó con éxito en una región específica del océano,
#  mientras que Océano Falso significa que la misión aterrizó sin éxito en una región específica del océano.
#  Verdadero RTLS significa que el resultado de la misión fue aterrizado con éxito 
# en una plataforma de tierra Falso RTLS significa que el resultado de la misión fue aterrizado 
# sin éxito en una plataforma de tierra Verdadero ASDS significa que el resultado de la misión fue aterrizado 
# con éxito en una nave no tripulada Falso ASDS significa que el resultado de la misión fue aterrizado sin éxito en una nave no tripulada.

#En este laboratorio convertiremos estos resultados en Etiquetas de Entrenamiento con 1 significa que el cohete aterrizó con éxito 
# y 0 significa que no tuvo éxito.

# Pandas es una biblioteca de software escrita para el lenguaje de programación Python para la manipulación y análisis de datos.
import pandas as pd
#NumPy es una biblioteca para el lenguaje de programación Python, que añade soporte para matrices y arrays multidimensionales de gran tamaño,
#  junto con una gran colección de funciones matemáticas de alto nivel para operar con estos arrays
import numpy as np

#Análisis de datos
# Cargar el conjunto de datos Space X, de la última sección.
df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
print(df.head(10))

#Identifique y calcule el porcentaje de los valores que faltan en cada atributo
df.isnull().sum()/len(df)*100

#Identifique qué columnas son numéricas y cuáles categóricas:
print(df.dtypes)

# TAREA 1: Calcular el número de lanzamientos en cada emplazamiento¶
# Los datos contienen varias instalaciones de lanzamiento de Space X: Cape Canaveral Space Launch Complex 40 VAFB SLC 4E ,
#  Vandenberg Air Force Base Space Launch Complex 4E (SLC-4E), Kennedy Space Center Launch Complex 39A KSC LC 39A .
# La ubicación de cada lanzamiento se coloca en la columna LaunchSite
# 
# A continuación, veamos el número de lanzamientos de cada emplazamiento.
# Utilice el método value_counts() en la columna LaunchSite para determinar el número de lanzamientos en cada sitio:
launch_counts = df['LaunchSite'].value_counts()
# Imprimir el número de lanzamientos en cada emplazamiento
print("Número de lanzamientos en cada emplazamiento:")
print(launch_counts)

# TAREA 2: Calcular el número y ocurrencia de cada órbita
# Utilice el método .value_counts() para determinar el número y ocurrencia de cada órbita en la columna Orbit
orbit_counts = df['Orbit'].value_counts()
# Imprimir el número y ocurrencia de cada órbita
print("\nNúmero y ocurrencia de cada órbita:")
print(orbit_counts)

#TAREA 3: Calcular el número y ocurrencia de resultados de misión de las órbitas

#Utiliza el método .value_counts() en la columna Outcome para determinar el número de landing_outcomes.
# Luego asígnalo a una variable landing_outcomes.
landing_outcomes = df['Outcome'].value_counts()
# Imprimir el número y ocurrencia de resultados de misión de las órbitas
print("\nNúmero y ocurrencia de resultados de misión de las órbitas:")
print(landing_outcomes)

#True Ocean significa que el resultado de la misión aterrizó con éxito en una región específica del océano,
#  mientras que False Ocean significa que el resultado de la misión aterrizó sin éxito en una región específica del océano.
#  Verdadero RTLS significa que la misión se aterrizó correctamente en una plataforma terrestre
#  Falso RTLS significa que la misión no se aterrizó correctamente en una plataforma terrestre
#  Verdadero ASDS significa que la misión se aterrizó correctamente en un dron 
# Falso ASDS significa que la misión no se aterrizó correctamente en un dron. 
# Ninguno ASDS y Ninguno Ninguno representan un fallo en el aterrizaje.
for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)
#Creamos un conjunto de resultados en los que la segunda etapa no aterrizó con éxito:
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
print(bad_outcomes)

#TAREA 4: Crear una etiqueta de resultado de aterrizaje a partir de la columna Resultado
#Usando el Resultado, crea una lista donde el elemento sea cero si la fila correspondiente en Resultado está en el conjunto mal_resultado;
#  en caso contrario, será uno. Luego asígnalo a la variable landing_class:
landing_class = [0 if outcome in bad_outcomes else 1 for outcome in df['Outcome']]
# Imprimir la etiqueta de resultado de aterrizaje
print("\nEtiqueta de resultado de aterrizaje:")
print(landing_class[:10])  # Imprimir los primeros 10 elementos de landing_class
# Añadir la etiqueta de resultado de aterrizaje al DataFrame
df['LandingClass'] = landing_class
# Imprimir las primeras filas del DataFrame con la nueva columna LandingClass
print("\nDataFrame con la nueva columna LandingClass:")
print(df.head(10))
#Podemos utilizar la siguiente línea de código para determinar la tasa de éxito:
success_rate = df['LandingClass'].mean() * 100
# Imprimir la tasa de éxito
print("\nTasa de éxito de los aterrizajes:")
print(f"{success_rate:.2f}%")
# TAREA 5: Calcular el número de lanzamientos exitosos y fallidos
# Utilizar el método value_counts() en la columna LandingClass para determinar el número de lanzamientos exitosos y fallidos    
landing_class_counts = df['LandingClass'].value_counts()
# Imprimir el número de lanzamientos exitosos y fallidos
print("\nNúmero de lanzamientos exitosos y fallidos:")
print(landing_class_counts)

df.to_csv("dataset_part_2.csv", index=False)