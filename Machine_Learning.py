# Objetivos
# 
# Realizar un Análisis Exploratorio de Datos y determinar Etiquetas de Entrenamiento
# 
# Crear una columna para la clase
# Estandarizar los datos
# Dividir en datos de entrenamiento y datos de prueba
# -Encontrar el mejor hiperparámetro para SVM, árboles de clasificación y regresión logística.
# 
# 
# Encontrar el método con mejor rendimiento utilizando datos de prueba

# Pandas es una biblioteca de software escrita para el lenguaje de programación Python para la manipulación y análisis de datos.
import pandas as pd
# NumPy es una librería para el lenguaje de programación Python, que añade soporte para matrices y arrays multidimensionales de gran tamaño, junto con una gran colección de funciones matemáticas de alto nivel para operar sobre estos arrays
import numpy as np
# Matplotlib es una librería de ploteo para python y pyplot nos proporciona un marco de ploteo similar a MatLab. Lo utilizaremos en nuestra función de trazado para trazar los datos.
import matplotlib.pyplot as plt
#Seaborn es una librería de visualización de datos de Python basada en matplotlib. Proporciona una interfaz de alto nivel para dibujar gráficos estadísticos atractivos e informativos
import seaborn as sns
# El preprocesamiento nos permite estandarizar nuestros datos
from sklearn import preprocessing
# Nos permite dividir nuestros datos en datos de entrenamiento y datos de prueba
from sklearn.model_selection import train_test_split
# Nos permite probar parámetros de algoritmos de clasificación y encontrar el mejor
from sklearn.model_selection import GridSearchCV
# Algoritmo de clasificación Logistic Regression
from sklearn.linear_model import LogisticRegression
# Algoritmo de clasificación Support Vector Machine
from sklearn.svm import SVC
# Algoritmo de clasificación Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Algoritmo de clasificación K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Esta función es para trazar la matriz de confusión.

def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 

# Cargar el marco de datos

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
data = pd.read_csv(url)
print(data.head(5))

URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
X = pd.read_csv(URL2)
print(X.head(5))

#TAREA 1
#Crea un array NumPy a partir de la columna Clase en datos, aplicando el método to_numpy() luego asígnalo a la variable Y,
#  asegúrate de que la salida es una serie Pandas (sólo un paréntesis df['nombre de la columna']).
Y = data['Class'].to_numpy()
# Verifica la forma de Y
print(Y.shape)
# Verifica el tipo de Y
print(type(Y))
# Verifica el tipo de datos de Y
print(Y.dtype)
# Verifica el número de clases en Y
print(np.unique(Y))

# TAREA 2
# Estandarice los datos en X y luego reasígnelos a la variable X utilizando la transformación proporcionada a continuación.
transform = preprocessing.StandardScaler()
# Dividimos los datos en datos de entrenamiento y de prueba mediante la función train_test_split.
#  Los datos de entrenamiento se dividen en datos de validación,
#  un segundo conjunto utilizado para los datos de entrenamiento; 
# a continuación, se entrenan los modelos y se seleccionan los hiperparámetros mediante la función GridSearchCV.


# TAREA 3
# Utilice la función train_test_split para dividir los datos X e Y en datos de entrenamiento y datos de prueba.
#  Establezca el parámetro test_size en 0,2 y random_state en 2. Los datos de entrenamiento y
#  los datos de prueba deben asignarse a las siguientes etiquetas.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# Estandarizar los datos
X_train = transform.fit_transform(X_train)
X_test = transform.transform(X_test)
# Verifica la forma de X_train
print(X_train.shape)
# Verifica la forma de X_test
print(X_test.shape)
# Verifica la forma de Y_train
print(Y_train.shape)
# Verifica la forma de Y_test
print(Y_test.shape)


# TAREA 4
# Cree un objeto de regresión logística y luego cree un objeto GridSearchCV logreg_cv con cv = 10.
#  Ajuste el objeto para encontrar los mejores parámetros a partir de los parámetros del diccionario.

# Definimos los parámetros para la regresión logística.
#  En este caso, estamos utilizando la regularización L2 con el solucionador 'lbfgs'.
#  Los parámetros son una lista de valores para el parámetro C, que controla la regularización.
#  Los valores de C son 0.01, 0.1 y 1
parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}
parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()
logreg_cv = GridSearchCV(LogisticRegression(), parameters, cv=10)
# Ajuste el objeto GridSearchCV a los datos de entrenamiento X_train e Y_train.
logreg_cv.fit(X_train, Y_train)
#  Obtenemos el objeto GridSearchCV para la regresión logística. Mostramos los mejores parámetros utilizando el atributo de datos
#  best_params_ y la precisión en los datos de validación utilizando el atributo de datos best_score_.
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

# TAREA 5
# Calcule la precisión en los datos de prueba utilizando la puntuación del método:
logreg_score = logreg_cv.score(X_test, Y_test)
print("Logistic Regression Test Accuracy: ", logreg_score)
# Veamos la matriz de confusión:
yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
# Examinando la matriz de confusión, vemos que la regresión logística puede distinguir entre las distintas clases.
#  Vemos que el problema son los falsos positivos.
# 
# Resumen:
# Verdadero positivo - 12 (La etiqueta verdadera está en el suelo, la etiqueta predicha también está en el suelo)
# 
# Falsos positivos - 3 (La etiqueta verdadera no se ha encontrado, la etiqueta predicha sí)

# TAREA 6
# Crear un objeto máquina de vectores soporte y luego crear un objeto GridSearchCV svm_cv con cv = 10.
#  Ajuste el objeto para encontrar los mejores parámetros a partir de los parámetros del diccionario.
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()
svm_cv = GridSearchCV(SVC(), parameters, cv=10)
# Ajuste el objeto GridSearchCV a los datos de entrenamiento X_train e Y_train.
svm_cv.fit(X_train, Y_train)
#  Obtenemos el objeto GridSearchCV para la máquina de vectores de soporte.
print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)

# TAREA 7
# Calcule la precisión en los datos de prueba utilizando la puntuación del método:
svm_score = svm_cv.score(X_test, Y_test)
print("SVM Test Accuracy: ", svm_score) 
# Veamos la matriz de confusión:
yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

#TAREA 8
# Cree un objeto clasificador de árbol de decisión y luego cree un objeto GridSearchCV tree_cv con cv = 10. 
# Ajuste el objeto para encontrar los mejores parámetros a partir de los parámetros del diccionario.
parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(DecisionTreeClassifier(), parameters, cv=10)
# Ajuste el objeto GridSearchCV a los datos de entrenamiento X_train e Y_train. 
tree_cv.fit(X_train, Y_train)
#  Obtenemos el objeto GridSearchCV para el árbol de decisión.
print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)

# TAREA 9
# Calcule la precisión de tree_cv en los datos de prueba utilizando la puntuación del método:
tree_score = tree_cv.score(X_test, Y_test)
print("Decision Tree Test Accuracy: ", tree_score)
# Veamos la matriz de confusión:
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# TAREA 10
# Cree un objeto k vecinos más cercanos y luego cree un objeto GridSearchCV knn_cv con cv = 10. 
# Ajuste el objeto para encontrar los mejores parámetros a partir de los parámetros del diccionario.
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()
knn_cv = GridSearchCV(KNeighborsClassifier(), parameters, cv=10)
# Ajuste el objeto GridSearchCV a los datos de entrenamiento X_train e Y_train.
knn_cv.fit(X_train, Y_train)
#  Obtenemos el objeto GridSearchCV para KNN.
print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)

# TAREA 11
# Calcule la precisión de knn_cv en los datos de prueba utilizando la puntuación del método:
knn_score = knn_cv.score(X_test, Y_test)
print("KNN Test Accuracy: ", knn_score)
# Veamos la matriz de confusión:
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

# TAREA 12
# Encuentra el método que mejor funcione:
scores = {'Logistic Regression': logreg_score,
          'SVM': svm_score,
          'Decision Tree': tree_score,
          'KNN': knn_score}
best_method = max(scores, key=scores.get)
print("Best method: ", best_method)
# Imprimir las puntuaciones de cada método
for method, score in scores.items():
    print(f"{method} Test Accuracy: {score:.4f}")
# Resumen de los resultados
print("\nSummary of Results:")
for method, score in scores.items():
    print(f"{method}: {score:.4f}")
# El mejor método es el que tiene la mayor precisión en los datos de prueba.
print(f"\nBest Method: {best_method} with accuracy {scores[best_method]:.4f}")
# Guardar el modelo con mejor rendimiento
import joblib
best_model = None
if best_method == 'Logistic Regression':
    best_model = logreg_cv.best_estimator_
elif best_method == 'SVM':
    best_model = svm_cv.best_estimator_
elif best_method == 'Decision Tree':
    best_model = tree_cv.best_estimator_
elif best_method == 'KNN':
    best_model = knn_cv.best_estimator_
joblib.dump(best_model, 'best_model.pkl')
print(f"Best model saved as 'best_model.pkl'")  
# Fin del script
print("End of script.")
