# Proyecto: Indice Automotores

## Visión general de la Aplicación

Se contruyó una app interactiva que permite emular un tablero para **visualizar la variación de precios de Repuestos y Mano de Obra en el sector automotriz**, 

Fue desarrollada en Python a partir de librerías como **Streamlit** para el desarrollo de la app y **Plotly** para la construcción de los gráficos. 


<img width="1912" height="966" alt="inicio_app" src="https://github.com/user-attachments/assets/8bf54ca4-e1c3-4718-bf11-fbba4f147f36" />


---

## Fuentes de datos

La aplicación se alimenta de **archivos CSV** con los datos necesarios para generar los gráficos. Estos datos provienen de  **dos fuentes** claves del sector, incluyendo:
* **Pilkington:** Listas de precios detalladas de cristales e instalación (parabrisas, lunetas, etc.).
* **Orion/Cesvi:** Información sobre costos de repuestos y mano de obra más relevantes del mercado.

---

## Características

La aplicación permite visualizar la evolución de costos en el tiempo a través de gráficos interactivos con diversos filtros.

Para repuestos de fuente **PILKINGTON**:
* **Precios de Material Históricos**
* **Costo de Instalación Histórico** 
* **Precios de Material (Ajustados por IPC)** 
* **Costo de Instalación (Ajustados por IPC)**

Para repuestos de fuente **ORION/CESVI**:
* **Costo de piezas prom. por TVA (Auto/Camión)**
* **Costo de piezas prom. por Tipo Repuesto**
* **Costo de piezas prom. histórico por Marca (autos)**
* **Costo de piezas prom. histórico por Marca (camiones)**
* **Costo de mano de obra prom. histórico por Tipo de M.O y TVA**
* **Comparativa variación costos M.O - CLEAS SI vs CLEAS NO**

Estas seis variaciones se analizan por valores ***históricos, ajustados por IPC y a valor dólar***.


---

## Despliegue y acceso

La aplicación está desplegada en la nube a través de **Streamlit Community Cloud**.

**Enlace para acceder a la app:** 

### [<ins>**indiceautomotores.streamlit.app**</ins>](https://indiceautomotores.streamlit.app/)

---

## Para ejecución local:
1. Descargar el repositorio clickeando en ``<> Code`` -> Download ZIP.
2. Extraer los archivos y abrir el repositorio en un editor de código (por ej. VS Code) 
3. Instalar las dependencias -> ejecutar en terminal:
```sh
pip install -r requirements.txt
```
4. Luego correr:
```sh
streamlit run app_streamlit.py
```
* si devuelve puerto ocupado o PermissionError
  
  -> probar con otros puertos, por ej.:
```sh
streamlit run app_streamlit.py --server.port 8502
```

Se abrirá la app localmente en su navegador.
