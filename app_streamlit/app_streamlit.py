import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # Necesario para go.Figure en caso de datos vacíos

ds_mo_ipc = 'df_mo_long_ipc.csv'
ds_mo = 'df_mo_long.csv'
ds_rep_ipc = 'df_long_ipc.csv'
ds_rep = 'df_long.csv'

try:
    df_mo_long_ipc = pd.read_csv(ds_mo_ipc)
    df_mo_long = pd.read_csv(ds_mo)
    df_long_ipc = pd.read_csv(ds_rep_ipc)
    df_long = pd.read_csv(ds_rep)

except FileNotFoundError:
    st.error("Error: Asegúrate de que los archivos CSV (df_mo_long_ipc.csv, df_mo_long.csv, df_long_ipc.csv, df_long.csv) estén en el mismo directorio que tu script.")
    st.stop() # Detiene la ejecución de la app si los archivos no se encuentran

# --- 2. Preprocesamiento de Datos (como en tu app de Dash) ---
for df_temp in [df_long, df_mo_long, df_long_ipc, df_mo_long_ipc]:
    df_temp['fecha'] = pd.to_datetime(df_temp['fecha'])
    if 'tipo_cristal' in df_temp.columns:
        # Limpiar el nombre de la columna para que sea más legible en la leyenda/facetas
        df_temp['tipo_cristal'] = df_temp['tipo_cristal'].astype(str).str.replace('_', ' ').str.title()
    # Asegurarse de que 'marca' y 'zona' sean string para evitar problemas de tipo en Plotly
    if 'marca' in df_temp.columns:
        df_temp['marca'] = df_temp['marca'].astype(str)
    if 'zona' in df_temp.columns:
        df_temp['zona'] = df_temp['zona'].astype(str)


# --- 3. Definir la Interfaz de Usuario (UI) de Streamlit ---

st.set_page_config(layout="wide") # Opcional: usa el ancho completo de la página

st.title('Variación de Precios de Cristales y Mano de obra por Marca y Zona')
st.markdown("#### _Fuente de datos: Listas de precios de Pilkington_")
st.markdown("---")

# Obtener las opciones únicas para el Dropdown de Zona
available_zones = sorted(df_long['zona'].unique().tolist())

with st.sidebar:
    st.header("Filtros") # Título para la barra lateral
    st.markdown("---")
    #st.markdown("##### _Seleccionar Zona:_") 
    selected_zone = st.selectbox(
        "Zona",
        options=available_zones,
        index=0 # Selecciona la primera zona por defecto
    )
    st.markdown("---") # Separador visual en la barra lateral

def create_filtered_plot(df_source, y_col, y_label):
    # Filtrar el DataFrame según la zona seleccionada
    df_filtered = df_source[
        (df_source['zona'] == selected_zone)
    ]
    
    if df_filtered.empty:
        fig = go.Figure().update_layout(
            title_text=f"No hay datos para '{selected_zone}'",
            height=400,
            font=dict(family="Arial", size=10),
            title_font_size=12
        )
        return fig

    # Crear el gráfico de líneas de Plotly Express
    fig = px.line(
        df_filtered,
        x='fecha',
        y=y_col,
        color='marca', # Un color para cada marca
        line_group='marca',
        facet_col='tipo_cristal', # Subplots por tipo de cristal
        #title='',
        labels={'fecha': '', y_col: y_label, 'marca': 'Marca', 'tipo_cristal': 'Tipo de Cristal'}
    )

    # Ajustes de layout del gráfico
    fig.update_layout(
        height=400, # Altura del subplot individual
        legend_title_text='Marca',
        font=dict(family="Arial", size=11),
        #title_font_size=12,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    # Ajustar el título de las facetas para que sean más legibles
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    return fig

# Los gráficos se apilarán verticalmente por defecto
st.subheader('1. Precios de Material históricos (Sin IVA)')
fig1 = create_filtered_plot(df_long, 'precio', 'Precio Sin IVA')
st.plotly_chart(fig1, use_container_width=True)

st.subheader('2. Costo de Instalación histórico (Sin IVA)')
fig2 = create_filtered_plot(df_mo_long, 'instalacion', 'Costo de Instalación')
st.plotly_chart(fig2, use_container_width=True)

st.subheader('3. Precios de Material (Ajustados por IPC)')
fig3 = create_filtered_plot(df_long_ipc, 'precio_ipc', 'Precio (IPC)')
st.plotly_chart(fig3, use_container_width=True)

st.subheader('4. Costo de Instalación (Ajustados por IPC)')
fig4 = create_filtered_plot(df_mo_long_ipc, 'instalacion_ipc', 'Costo de Instalación (IPC)')
st.plotly_chart(fig4, use_container_width=True)






