#Objetivos
#Web scrap Falcon 9 launch records with BeautifulSoup:

#Extraer una tabla HTML de registros de lanzamiento de Falcon 9 de Wikipedia
#Analizar la tabla y convertirla en un marco de datos Pandas

import sys

import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import pandas as pd

# y le proporcionaremos algunas funciones de ayuda para que pueda procesar la tabla HTML raspada de la web
def date_time(table_cells):
    """
    This function returns the data and time from the HTML  table cell
    Input: the  element of a table data cell extracts extra row
    """
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    """
    This function returns the booster version from the HTML  table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=''.join([booster_version for i,booster_version in enumerate( table_cells.strings) if i%2==0][0:-1])
    return out

def landing_status(table_cells):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=[i for i in table_cells.strings][0]
    return out


def get_mass(table_cells):
    mass=unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass=mass[0:mass.find("kg")+2]
    else:
        new_mass=0
    return new_mass


def extract_column_from_header(row):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    if (row.br):
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()
        
    colunm_name = ' '.join(row.contents)
    
    # Filter the digit and empty names
    if not(colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name    
# Para que las tareas de laboratorio sean coherentes, se te pedirá que extraigas 
# los datos de una instantánea de la página Wikipage List of Falcon 9 and Falcon Heavy launches actualizada el 9 de junio de 2021.
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"

#TAREA 1: Solicitar la página Wiki Falcon9 Launch desde su URL
# En primer lugar, vamos a realizar un método HTTP GET para solicitar la página Falcon9 Launch HTML, como una respuesta HTTP.

# use requests.get() method with the provided static_url
# assign the response to a object
response = requests.get(static_url)

if response.status_code != 200:
    print(f"Error: Unable to fetch the page. Status code: {response.status_code}")
    sys.exit(1)
# assign the response to a object
soup = BeautifulSoup(response.content, 'html.parser')

# Use BeautifulSoup() to create a BeautifulSoup object from a response text content

# Use soup.title attribute
print(f"Page title: {soup.title.string}")

# TAREA 2: Extraer todos los nombres de columnas/variables de la cabecera de la tabla HTML
# A continuación, queremos recopilar todos los nombres de columnas relevantes de la cabecera de la tabla HTML.
# Intentemos encontrar primero todas las tablas de la página wiki. 
# Si necesitas refrescar tu memoria sobre BeautifulSoup,
#  por favor revisa el enlace de referencia externa hacia el final de este laboratorio.

# Use the find_all function in the BeautifulSoup object, with element type `table`
tables = soup.find_all('table')


# Assign the result to a list called `html_tables`
html_tables = []
for table in tables:
    if table.get('class') and 'wikitable' in table.get('class'):
        html_tables.append(table)
# Check if we have found any tables
if not html_tables:
    print("No tables found with class 'wikitable'.")
    sys.exit(1)
# Print the number of tables found
print(f"Number of tables found: {len(html_tables)}")

# A partir de la tercera tabla es nuestra tabla de destino contiene los registros de lanzamiento real.
# Let's print the third table and check its content
first_launch_table = html_tables[2]
print(first_launch_table)

#A continuación, sólo tenemos que iterar a través de los elementos <th> y aplicar la función extract_column_from_header()
#  proporcionada para extraer el nombre de la columna uno a uno

column_names = []

# Apply find_all() function with `th` element on first_launch_table
th_elements = first_launch_table.find_all('th')

# Iterate each th element and apply the provided extract_column_from_header() to get a column name
for th in th_elements:
    name = extract_column_from_header(th)
    # Append the Non-empty column name (`if name is not None and len(name) > 0`) into a list called column_names
    if name is not None and len(name) > 0:
        column_names.append(name)
# Print the column names
print(f"Column names: {column_names}")

# TAREA 3: Crear un marco de datos parseando las tablas HTML de lanzamiento
# Crearemos un diccionario vacío con las claves de los nombres de columna extraídos en la tarea anterior.
#  Posteriormente, este diccionario se convertirá en un marco de datos Pandas
launch_dict= dict.fromkeys(column_names)

# Remove an irrelvant column
del launch_dict['Date and time ( )']

# Let's initial the launch_dict with each value to be an empty list
launch_dict['Flight No.'] = []
launch_dict['Launch site'] = []
launch_dict['Payload'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
# Added some new columns
launch_dict['Version Booster']=[]
launch_dict['Booster landing']=[]
launch_dict['Date']=[]
launch_dict['Time']=[]

#A continuación, sólo tenemos que rellenar launch_dict con los registros de lanzamiento extraídos de las filas de la tabla.
# 
# Normalmente, las tablas HTML de las páginas Wiki suelen contener anotaciones inesperadas y otros tipos de ruidos,
#  como enlaces de referencia B0004.1[8], valores que faltan N/A [e], formato incoherente, etc.
# 
# Para simplificar el proceso de análisis sintáctico, 
# le ofrecemos a continuación un fragmento de código incompleto que le ayudará a rellenar el launch_dict. 
# Por favor, complete el siguiente fragmento de código con TODOs o puede optar por escribir su propia lógica
#  para analizar todas las tablas de lanzamiento:
extracted_row = 0
#Extract each table 
for table_number,table in enumerate(soup.find_all('table',"wikitable plainrowheaders collapsible")):
   # get table row 
    for rows in table.find_all("tr"):
        if rows.th and rows.th.string:
            flight_number = rows.th.string.strip()
            if flight_number.isdigit():
                row = rows.find_all('td')
                if len(row) < 9:  # Evita errores si la fila está incompleta
                    continue

                datatimelist = date_time(row[0])
                date = datatimelist[0].strip(',') if len(datatimelist) > 0 else ''
                time = datatimelist[1] if len(datatimelist) > 1 else ''

                bv = booster_version(row[1]) or (row[1].a.string if row[1].a else '')

                launch_site = row[2].a.string if row[2].a else row[2].text.strip()
                payload = row[3].a.string if row[3].a else row[3].text.strip()
                payload_mass = get_mass(row[4])
                orbit = row[5].a.string if row[5].a else row[5].text.strip()
                customer = row[6].a.string if row[6].a else row[6].text.strip()
                launch_outcome = list(row[7].strings)[0] if row[7].strings else ''
                booster_landing = landing_status(row[8])

            # Agregar cada valor al diccionario
                launch_dict['Flight No.'].append(flight_number)
                launch_dict['Date'].append(date)
                launch_dict['Time'].append(time)
                launch_dict['Version Booster'].append(bv)
                launch_dict['Launch site'].append(launch_site)
                launch_dict['Payload'].append(payload)
                launch_dict['Payload mass'].append(payload_mass)
                launch_dict['Orbit'].append(orbit)
                launch_dict['Customer'].append(customer)
                launch_dict['Launch outcome'].append(launch_outcome)
                launch_dict['Booster landing'].append(booster_landing)

df = pd.DataFrame({key: pd.Series(value) for key, value in launch_dict.items()})
df.to_csv('spacex_web_scraped.csv', index=False)


# Pregunta 1
# Después de realizar una solicitud GET en la API de Space X y convertir la respuesta a un marco de datos utilizando pd.json_normalize. 
# ¿Qué año se encuentra en la primera fila de la columna static_fire_date_utc?
static_json_url = "https://api.spacexdata.com/v4/launches/past"
response = requests.get(static_json_url)
data = pd.json_normalize(response.json())
first_row_date = pd.to_datetime(data['static_fire_date_utc'].iloc[0])
print(f"Year in the first row of 'static_fire_date_utc': {first_row_date.year}")
#Pregunta 2
#Utilizando la API, ¿cuántos lanzamientos de Falcon 9 hay después de eliminar los de Falcon 1?
launches = data[data['rocket'].str.contains('Falcon 9')]
num_falcon9_launches = launches.shape[0]    
print(f"Number of Falcon 9 launches: {num_falcon9_launches}")

# Pregunta 3
# Al final del proceso de recogida de datos de la API , ¿cuántos valores que faltan hay para la columna landingPad?
missing_landing_pad = df['Booster landing'].isnull().sum()
print(f"Number of missing values in 'Booster landing': {missing_landing_pad}")

#Pregunta 4
# Tras realizar una petición a la página Wiki del lanzamiento del Falcon9 y crear un objeto BeautifulSoup, ¿cuál es la salida de soup.title

page_title = soup.title.string
print(f"Page title: {page_title}")  
