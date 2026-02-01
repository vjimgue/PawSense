# PawSense

PawSense es una aplicación que utiliza la inteligencia artificial para predecir, con la mayor precisión posible, la mezcla de razas de tu perro. 

Es una herramienta pensada tanto para organizaciones como protectoras, refugios y veterinarias, donde puede ser útil para entender mejor los cuidados de un perro e incluso prevenir futuras enfermedades a las que pueden ser propensos, como para usuarios particulares que tienen curiosidad por saber las razas y características que estas implican para su mascota.
## Alumnos:
- Victor Jiménez Guerrero
- Enrique Moreno Alcántara
- Carlos Cerezo López

## Dataset:
Para llevar a cabo este proyecto hemos utilizado la base de datos de Kaggle 'Stanford Dogs Dataset' que contiene mas de 20,000 imágenes de 120 razas.

Enlace a dataset de Kaggle: <https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset>


### Instrucciones para correr el contenedor:
Contenedor de desarrollo con jupyter notebook, streamlit y tensorflow instalado.
```
cd Dockerfile
docker compose build
docker compose up
```
Crear una carpeta llamada dataset_perros y descomprimir el dataset en esta carpeta, para poder generar las imagenes recortadas.

En caso de usar el cuderno con PyTorch, crear un entorno virtual de python con los requirements.txt del directorio principal.

### Aplicación de streamlit:
Enlace a la aplicación en vivo: <https://pawsense.streamlit.app/> 

Si bien en esta rama se encuentra todo el codigo de la aplicación, se ha creado una rama aparte dedicada a streamlit con tal de evitar conflictos a la hora de hacer el deployment con un requirements.txt especifico para la aplicación de streamlit.
<img src="Images/CapturaWeb.gif" width=700>

