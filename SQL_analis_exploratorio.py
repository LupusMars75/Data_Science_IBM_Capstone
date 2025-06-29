#Visión general del conjunto de datos
# SpaceX ha acaparado la atención mundial por una serie de hitos históricos.

#Es la única empresa privada que ha conseguido devolver una nave espacial desde una órbita terrestre baja, lo que logró por primera vez en diciembre de 2010. SpaceX anuncia en su página web lanzamientos de cohetes Falcon 9 con un coste de 62 millones de dólares mientras que otros proveedores cuestan más de 165 millones de dólares cada uno, gran parte del ahorro se debe a que Space X puede reutilizar la primera etapa.

#Por lo tanto, si podemos determinar si la primera etapa aterrizará, podemos determinar el coste de un lanzamiento.

#Esta información puede utilizarse si una empresa alternativa quiere competir con SpaceX por el lanzamiento de un cohete.

#Este conjunto de datos incluye un registro para cada carga útil transportada durante una misión de SpaceX al espacio exterior.







#Descargue los conjuntos de datos
# Esta tarea requiere que cargue el conjunto de datos de SpaceX.

#En muchos casos, el conjunto de datos que se va a analizar está disponible como archivo .CSV (valores separados por comas), quizás en Internet. Haga clic en el siguiente enlace para descargar y guardar el conjunto de datos (archivo .CSV):

#Conjunto de datos Spacex


import csv, sqlite3
import prettytable
prettytable.DEFAULT = 'DEFAULT'

con = sqlite3.connect("my_data1.db")
cur = con.cursor()

import pandas as pd
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")

# Tareas
# Ahora escriba y ejecute consultas SQL para resolver las tareas asignadas.
# Nota: Si los nombres de columna están en mayúsculas y minúsculas, enciérrelos entre comillas dobles.
# Tarea 1
# Mostrar los nombres de los puntos de lanzamiento únicos en la misión espacial
# #Descargue los conjuntos de datos
# Esta tarea requiere que cargue el conjunto de datos de SpaceX.
cur.execute("SELECT DISTINCT LAUNCH_SITE FROM SPACEXTBL")
rows = cur.fetchall()
print("Tarea 1: Nombres de los puntos de lanzamiento únicos en la misión espacial")
for row in rows:
    print(row[0])
#Tarea 2
# Mostrar 5 registros en los que las bases de lanzamiento empiecen por la cadena 'CCA'
cur.execute("SELECT * FROM SPACEXTBL WHERE LAUNCH_SITE LIKE 'CCA%' LIMIT 5")
rows = cur.fetchall()   
print("\nTarea 2: Registros en los que las bases de lanzamiento empiecen por la cadena 'CCA'")
for row in rows:
    print(row)

#Tarea 3
# Mostrar la masa total de carga útil transportada por los boosters lanzados por la NASA (CRS)
cur.execute("SELECT SUM(PAYLOAD_MASS__KG_) FROM SPACEXTBL WHERE CUSTOMER = 'NASA (CRS)'")
rows = cur.fetchall()
print("\nTarea 3: Masa total de carga útil transportada por los boosters lanzados por la NASA (CRS)")
for row in rows:
    print(row[0])

# Tarea 4
# Visualizar la masa media de la carga útil transportada por la versión F9 v1.1 del booster.
cur.execute("SELECT AVG(PAYLOAD_MASS__KG_) FROM SPACEXTBL WHERE BOOSTER_VERSION = 'F9 v1.1'")
rows = cur.fetchall()
print("\nTarea 4: Masa media de la carga útil transportada por la versión F9 v1.1 del booster")
for row in rows:
    print(row[0])

# Tarea 5
#Indique la fecha en la que se produjo el primer aterrizaje con éxito en la plataforma.
# Sugerencia:Utilice la función min
cur.execute("SELECT MIN(DATE) FROM SPACEXTBL WHERE LANDING_OUTCOME = 'Success (ground pad)'")
rows = cur.fetchall()
print("\nTarea 5: Fecha del primer aterrizaje con éxito en la plataforma")
for row in rows:
    print(row[0])

# Tarea 6
# Enumere los nombres de los propulsores que han tenido éxito en la nave no tripulada y tienen una masa de carga útil superior a 4000 pero inferior a 6000.
cur.execute("SELECT BOOSTER_VERSION FROM SPACEXTBL WHERE LANDING_OUTCOME = 'Success (ground pad)' AND PAYLOAD_MASS__KG_ > 4000 AND PAYLOAD_MASS__KG_ < 6000")
rows = cur.fetchall()
print("\nTarea 6: Nombres de los propulsores con éxito en la nave no tripulada y masa de carga útil entre 4000 y 6000")
for row in rows:
    print(row[0])   

# Tarea 7
# Enumere el número total de misiones realizadas con éxito y fracasadas que incluyan la palabra Success o Failure en el resultado del aterrizaje.
cur.execute("SELECT LANDING_OUTCOME, COUNT(*) FROM SPACEXTBL WHERE LANDING_OUTCOME LIKE '%Success%' OR LANDING_OUTCOME LIKE '%Failure%' GROUP BY LANDING_OUTCOME")
rows = cur.fetchall()   
print("\nTarea 7: Número total de misiones realizadas con éxito y fracasadas")
for row in rows:
    print(f"{row[0]}: {row[1]} misiones")   

# tarea 8
#  Enumere todas las booster_versions que han transportado la masa máxima de carga útil, 
# utilizando una subconsulta con una función agregada adecuada. 
cur.execute("""
SELECT BOOSTER_VERSION FROM SPACEXTBL
WHERE PAYLOAD_MASS__KG_ = (
    SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEXTBL
)
""")
rows = cur.fetchall()
print("\nTarea 8: Booster_versions que han transportado la masa máxima de carga útil")
for row in rows:
    print(row[0])

# Tarea 9
# Enumere los registros que mostrarán los nombres de los meses, los resultados del aterrizaje fallido en la nave no tripulada,
#  las versiones de los propulsores y el lugar de lanzamiento para los meses del año 2015.
# Nota: SQLLite no admite nombres de mes.
#  Por lo tanto, debe utilizar substr(Date,6,2) como mes para obtener los meses y substr(Date,0,5)=“2015” para el año.
cur.execute("""
SELECT substr(DATE, 6, 2) AS MONTH, LANDING_OUTCOME, BOOSTER_VERSION, LAUNCH_SITE
FROM SPACEXTBL  
WHERE LANDING_OUTCOME LIKE '%Failure%' AND substr(DATE, 0, 5) = '2015'
""")
rows = cur.fetchall()
print("\nTarea 9: Registros de meses, resultados de aterrizaje fallidos, versiones de propulsores y lugar de lanzamiento para los meses de 2015")
for row in rows:
    print(f"Mes: {row[0]}, Resultado de aterrizaje: {row[1]}, Versión del propulsor: {row[2]}, Lugar de lanzamiento: {row[3]}")

#Tarea 10
# Clasifique el recuento de resultados de aterrizaje (como Fracaso (nave no tripulada) o Éxito (plataforma en tierra)) 
# entre la fecha 2010-06-04 y 2017-03-20, en orden descendente.
cur.execute("""
SELECT LANDING_OUTCOME, COUNT(*) AS COUNT
FROM SPACEXTBL
WHERE DATE BETWEEN '2010-06-04' AND '2017-03-20'
GROUP BY LANDING_OUTCOME
ORDER BY COUNT DESC
""")
rows = cur.fetchall()
print("\nTarea 10: Recuento de resultados de aterrizaje entre 2010-06-04 y 2017-03-20")
for row in rows:
    print(f"Resultado de aterrizaje: {row[0]}, Recuento: {row[1]}")

# Cierre la conexión a la base de datos
con.close() 
