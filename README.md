# Proyecto: Indice Automotores

## Visión General de la Aplicación

Se contruyó una app interactiva, desarrollada con **Streamlit** y **Plotly**, para **visualizar la variación de precios de Repuestos y Mano de Obra en el sector automotriz**. 

<img width="1908" height="996" alt="image" src="https://github.com/user-attachments/assets/1f87c64a-bd40-475e-b9c7-b5f4a4197271" />



---

## Fuentes de Datos

La aplicación se alimenta de **cuatro archivos CSV** con los datos necesarios para generar los gráficos. Estos datos provienen de fuentes clave del sector, incluyendo:
* **Pilkington:** Listas de precios detalladas de cristales (parabrisas, lunetas, etc.).
* **Cesvi/Orion:** Información relevante sobre costos de repuestos, mano de obra y otros factores del mercado.

---

## Características

La aplicación permite visualizar la evolución de costos en el tiempo a través de cuatro gráficos interactivos con diversos filtros:
* **Precios de Material Históricos:** Evolución de los precios de los repuestos.
* **Costo de Instalación Histórico:** Evolución de los costos de mano de obra.
* **Precios de Material (Ajustados por IPC):** Variación de precios de repuestos considerando la inflación.
* **Costo de Instalación (Ajustados por IPC):** Variación de costos de mano de obra considerando la inflación.

Los usuarios pueden **seleccionar la zona** de interés (ej. CABA, Rosario) mediante un menú desplegable en la barra lateral, y todos los gráficos se actualizarán automáticamente para reflejar los datos de esa región. Cada gráfico, a su vez, permite observar la evolución de diferentes **tipos de cristal** y **marcas** de vehículos.

---

## Despliegue y Acceso

La aplicación está desplegada en la nube a través de **Streamlit Community Cloud**.

**Enlace para acceder a la app:** 

### [<ins>**indiceautomotores.streamlit.app**</ins>](https://indiceautomotores.streamlit.app/)

---

## Para Ejecución Local:
1. Descargar el repositorio clickeando en ``<> Code`` -> Download ZIP.
2. Extraer los archivos y abrir el repositorio en el editor de código (por ej. VS Code) 
3. Instalar las dependencias -> ejecutar `pip install -r requirements.txt` en la terminal
4. En la terminal correr: `streamlit run app_streamlit.py`
* si devuelve puerto ocupado o PermissionError
  
  -> probar con otros puertos, por ej.:
  `streamlit run app_streamlit.py --server.port 8502`

Se abrirá la app localmente en su navegador.
