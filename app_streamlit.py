import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # Necesario para go.Figure en caso de datos vacíos
import json
import requests
from io import StringIO
import unicodedata

pd.options.display.max_columns=None
pd.set_option('display.max_rows', 500)
pd.options.display.float_format = '{:,.2f}'.format

# ---------------------------------------------------

st.set_page_config(
    page_title="Indice Automotores ",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

"""
# :material/query_stats: Indice Automotores
"""

# ----- Cargo las bases de datos --------------------------------------------------
try:
    # df PKT (cristales)
    df_cristal = pd.read_csv('data/base_pkt_app_2.csv')

    # dfs de repuestos orion/cesvi
    df_tipo_rep = pd.read_csv('data/df_tipo_rep_sept.csv')
    df_rep_tv = pd.read_csv('data/df_rep_tv_sept.csv')

    # dfs de mano de obra orion/cesvi
    df_cm_mo_hist = pd.read_csv('data/df_cm_mo_hist.csv')
    df_cm_mo_ipc = pd.read_csv('data/df_cm_mo_ipc.csv')
    df_cm_mo_usd = pd.read_csv('data/df_cm_mo_usd.csv')
    # dfs mano de obra cleas si/cleas no
    df_cm_mo_hist_cleas = pd.read_csv('data/df_cm_mo_hist_cleas.csv')
    df_cm_mo_ipc_cleas = pd.read_csv('data/df_cm_mo_ipc_cleas.csv')
    df_cm_mo_usd_cleas = pd.read_csv('data/df_cm_mo_usd_cleas.csv')

    # dfs marcas
    df_marcas_autos = pd.read_csv('data/evol_todas_marcas_autos.csv')
    df_rtos_marca_mes = pd.read_csv('data/df_rtos_marca_mes_sept.csv')
    df_marcas_camiones = pd.read_csv('data/camion_marcas.csv')
    df_rtos_marca_mes_cam = pd.read_csv('data/df_rtos_marca_mes_cam_sept.csv')
    df_distrib_marcas_cartera = pd.read_csv('data/distrib_ar_marca_cartera.csv')

    # dfs p/mostrar tablas
    df_cm_mo_resumen = pd.read_csv('data/df_cm_mo_resumen.csv')
    df_cm_mo_cleas_resumen = pd.read_csv('data/df_cm_mo_cleas_resumen.csv')

    # dfs var x prov
    df_cm_prov_orion = pd.read_csv('data/base_cm_prov_orion.csv')
    df_cm_prov = pd.read_csv('data/base_cm_prov_ok.csv')
    with open('data/prov.geojson', 'r', encoding='utf-8') as f:
        provincias_geojson = json.load(f)
    comparativo_orion_prov = pd.read_csv('data/comparativo_orion_prov.csv')
    comparativo_cm_siniestral = pd.read_csv('data/comparativo_cm_siniestral.csv')

    # dfs comparativo mano de obra
    df_chapa_pintura = pd.read_csv('data/df_chapa_pintura.csv')
    df_mo_repuestos_final = pd.read_csv('data/df_mo_repuestos_final.csv')
    df_peritaciones = pd.read_csv('data/df_peritaciones.csv')
    df_costo_hora = pd.read_csv('data/df_costo_hora.csv')
    df_tot_reparacion = pd.read_csv('data/df_tot_reparacion.csv')

except FileNotFoundError as e:
    st.error(f"Error: No se encuentra el archivo CSV. Detalles: {e}")
    # La app se detiene si no encuentra los archivos
    st.stop()

# ---- Formateo de datos --------------------------------------------------
for df_temp in [df_cristal]:
    if 'fecha' in df_temp.columns:
        df_temp['fecha'] = pd.to_datetime(df_temp['fecha'])
    if 'año_mes' in df_temp.columns:
        df_temp['año_mes'] = pd.to_datetime(df_temp['año_mes'], format='%Y-%m-%d')
    if 'cristal' in df_temp.columns:
        df_temp['cristal'] = df_temp['cristal'].astype(str).str.replace('_', ' ').str.title()
    if 'marca' in df_temp.columns:
        df_temp['marca'] = df_temp['marca'].astype(str)
    if 'zona' in df_temp.columns:
        df_temp['zona'] = df_temp['zona'].astype(str)
    if 'tipo_repuesto' in df_temp.columns:
        df_temp['tipo_repuesto'] = df_temp['tipo_repuesto'].astype(str).str.replace('_', ' ').str.title()

# ---- Formateo base provincias --------------------------------------------------
df_cm_agg = df_cm_prov.groupby(['coverable','año','provincia',]).agg(
    coste_medio_prom=('coste_medio', 'mean'))
df_cm_agg = df_cm_agg.reset_index()
# cambio formato de coste medio a int
df_cm_agg['coste_medio_prom'] = df_cm_agg['coste_medio_prom'].astype(int)


# ---- Función construir graficos de torta -------------------------------------------------- 
def create_pie_chart(df, value):
    fig = px.pie(
    df,
    values=value,
    names='marca_agrupada',
    hover_data=[value],
    color_discrete_sequence=px.colors.qualitative.G10,
    labels={value: value.title(), 'marca_agrupada': 'Marca'},
    hole=0.3 # Agrega un agujero al centro para un estilo de "donut chart"
    )

    fig.update_traces(
        # textposition='inside',
        textinfo='percent+label',
        insidetextfont=dict(size=12, color='white', family='Arial')
    )
    fig.update_layout(
        font=dict(family="Arial", size=12, color="white"),
        showlegend=True
    )

    return fig 

# ---- Slider selección de análisis --------------------------------------------------
st.markdown("---")
st.markdown("### Seleccionar el análisis deseado:")
selected_analysis = st.selectbox(
    'Seleccionar Análisis:',
    options=["Evolutivo precios Pilkington", 
             "Evolutivo precios ORION/CESVI",
             "Análisis por Provincia",
             "Comparativo de Mano de Obra (L2/Cesvi)"],
    index=0,
    label_visibility ='collapsed'
)
st.markdown("---")

# ---- Variables de sesión para mostrar/ocultar gráficos --------------------------------------------------
if 'show_pie_chart' not in st.session_state:
    st.session_state.show_pie_chart = False

if 'show_pie_chart_2' not in st.session_state:
    st.session_state.show_pie_chart_2 = False

if 'show_pie_chart_3' not in st.session_state:
    st.session_state.show_pie_chart_3 = False

if 'show_mo' not in st.session_state:
    st.session_state.show_mo = False

# ---- Análisis PILKINGTON --------------------------------------------------
if selected_analysis == "Evolutivo precios Pilkington":
    st.title("Variación de Precios de Cristales y Mano de obra por Marca y Zona")
    st.markdown("#### _Fuente de datos: Listas de precios de Pilkington_")
    st.markdown("---")

    # Dropdown de Zona (barra lateral)
    available_zones = sorted(df_cristal['zona'].unique().tolist())
    available_marcas = sorted(df_cristal['marca'].unique().tolist())
    DEFAULT_MARCAS = ["TOYOTA", "VOLKSWAGEN", "FORD", "CHEVROLET", "PEUGEOT", "RENAULT", "FIAT"]

    with st.sidebar:
        # st.header("Filtros") # Título para la barra lateral
        st.markdown("---")
        st.markdown("##### _Seleccionar Zona:_") 
        selected_zone = st.selectbox(
            "Zona",
            options=available_zones,
            index=0, # primera zona por defecto,
            label_visibility ='collapsed'
        )
        st.markdown("---")
        st.session_state['zona'] = selected_zone
        st.markdown("##### _Seleccionar Marcas:_")
        selected_marcas = st.multiselect(
            "Marcas",
            options=available_marcas,
            default=[m for m in DEFAULT_MARCAS if m in available_marcas],
            label_visibility='collapsed',
        )
        st.markdown("---")

    def create_plot_pkt(df_source, y_col, y_label):   

        # Filtrar el DataFrame según la zona seleccionada
        df_filtered = df_source[
            (df_source['zona'] == selected_zone) &
            (df_source['marca'].isin(selected_marcas))
        ]

        # gráfico Plotly 
        fig = px.line(
            df_filtered,
            x='fecha',
            y=y_col,
            color='marca', # Un color para cada marca
            line_group='marca',
            facet_col='cristal', # Subplots por tipo de cristal
            labels={'fecha': '', y_col: y_label, 'marca': 'Marca', 'cristal': 'Tipo de Cristal'}
        )

        # Ajustes del gráfico
        fig.update_layout(
            height=400, # Altura del subplot individual
            legend_title_text='Marca',
            font=dict(family="Arial", size=15),
            margin=dict(t=50, b=0, l=0, r=0)
        )
        # Ajustar el título de las facetas para que sean más legibles
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        return fig

    # ----- GRAFICOS HISTORICOS, IPC y USD --------------------------------------------------

    if not selected_marcas:
        st.warning("Seleccionar una marca para ver la información.")
    else:
        # muestro grafico torta MARCAS AUTOS 
        if st.button("Mostrar/Ocultar Distribución de Marcas Autos"):
            st.session_state.show_pie_chart_3 = not st.session_state.show_pie_chart_3
        
        if st.session_state.show_pie_chart_3:
            st.subheader('Distribución de Años Riesgo por Marca')
            fig_pie = create_pie_chart(df_distrib_marcas_cartera, 'años_riesgos_total')
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('Total AR: ' + str(df_distrib_marcas_cartera['años_riesgos_total'].sum()))
            st.markdown('Total marcas: 49' )
            st.markdown("---")

        with st.container(border=True):
            st.subheader('1. Precios de Material históricos (Sin IVA)')
            fig1 = create_plot_pkt(df_cristal, 'precio', 'Precio Sin IVA')
            st.plotly_chart(fig1, use_container_width=True)

        with st.container(border=True):
            st.subheader('2. Costo de Instalación histórico (Sin IVA)')
            fig2 = create_plot_pkt(df_cristal, 'instalacion', 'Costo de Instalación')
            st.plotly_chart(fig2, use_container_width=True)

        with st.container(border=True):
            st.subheader('3. Precios de Material (Ajustados por IPC)')
            fig3 = create_plot_pkt(df_cristal, 'precio_ipc', 'Precio (IPC)')
            st.plotly_chart(fig3, use_container_width=True)

        with st.container(border=True):
            st.subheader('4. Costo de Instalación (Ajustados por IPC)')
            fig4 = create_plot_pkt(df_cristal, 'instalacion_ipc', 'Costo de Instalación (IPC)')
            st.plotly_chart(fig4, use_container_width=True)

        with st.container(border=True):
            st.subheader('5. Precios de Material (USD)')
            fig5 = create_plot_pkt(df_cristal, 'precio_usd', 'Precio (USD)')
            st.plotly_chart(fig5, use_container_width=True)

        with st.container(border=True):
            st.subheader('6. Costo de Instalación (USD)')
            fig6 = create_plot_pkt(df_cristal, 'instalacion_usd', 'Costo de Instalación (USD)')
            st.plotly_chart(fig6, use_container_width=True)     
        ""
        ""
        st.markdown("#### Data Cruda")

        df_filtered_raw = df_cristal[
            (df_cristal['zona'] == selected_zone) &
            (df_cristal['marca'].isin(selected_marcas))
        ].copy()
        df_filtered_raw['fecha'] = df_filtered_raw['fecha'].dt.strftime('%Y-%m-%d')
        st.dataframe(df_filtered_raw, use_container_width=True)


# ---- Análisis ORION/CESVI --------------------------------------------------
elif selected_analysis == "Evolutivo precios ORION/CESVI":
    st.title('Variación de Precios de Repuestos y Mano de obra')
    st.markdown("#### _Fuente de datos: Orion/Cesvi_")
    st.markdown("---")

    # sidebar por tipo de variación: histórico, ipc, usd
    with st.sidebar:
        st.markdown("---")
        st.markdown("##### _Seleccionar Tipo de Variación:_")
        selected_variation_type = st.selectbox(
            "Tipo de Variación",
            options=["Histórico", "IPC", "USD"],
            index=0,
            label_visibility='collapsed'
        )
        st.markdown("---")
        # Guardo la selección en session_state para que la funcion pueda usarla
        st.session_state['selected_variation_type'] = selected_variation_type
    
    def create_plot_orion(df, y_col, color, facet_col, y_label, x_tickangle=None):       
        if df.empty:
            fig = go.Figure().update_layout(
                title_text=f"No hay datos para graficar",
                height=400,
                font=dict(family="Arial", size=10),
                title_font_size=12
            )
            return fig

        fig = px.line(
            df,
            x='año_mes',
            y=y_col,
            color=color,
            color_discrete_sequence=["#FB0D0D", "lightgreen", "blue", "gray", "magenta", "cyan", "orange", '#2CA02C'],
            facet_col=facet_col,
            # line_group='marca',
            # facet_col='tipo_cristal', # Subplots por tipo de cristal
            #title='', agrego titulo con subheader
            labels={'año_mes': '', y_col: y_label,}# 'marca': 'Marca', 'tipo_cristal': 'Tipo de Cristal'}
        )

        fig.update_layout(
            height=400, # Altura del subplot individual
            font=dict(family="Arial", size=15),
            margin=dict(t=50, b=0, l=0, r=0),
            xaxis_tickangle=x_tickangle
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        return fig
    
    
    # ----- GRAFICOS HISTORICOS --------------------------------------------------
    if st.session_state['selected_variation_type'] == "Histórico":
        
        # gráfico 1: evolución costo repuestos por tva
        st.subheader('1. Costo de piezas prom. histórico por TVA')

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_rep_tv[['tva','año_mes','cant_ocompra','cant_piezas_total','var_cant_piezas',
                                    'cant_piezas_prom','costo_pieza_prom_hist','var_costo_prom','monto_total_compras']], hide_index=True,)

        fig5 = create_plot_orion(df_rep_tv, 'costo_pieza_prom_hist', 'tva', None,'Costo Promedio')
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("---")

        # GRAFICO 2: evolución costo repuestos por tipo repuesto
        st.subheader('2. Costo de piezas prom. histórico por Tipo Repuesto')
        # muestro distribución MARCA AUTOS
        with st.expander("Ver tabla de datos",):
            st.dataframe(df_tipo_rep[['tipo_repuesto','año_mes', 'cant_ocompra', 'cant_piezas_total',
                        'cant_piezas_prom', 'costo_pieza_prom_hist', 'var_costo_prom',
                        'monto_total_compras']], hide_index=True)

        fig6 = create_plot_orion(df_tipo_rep, 'costo_pieza_prom_hist', 'tipo_repuesto', None,'Costo Promedio')
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("---")

        # muestro grafico torta MARCAS AUTOS 
        if st.button("Mostrar/Ocultar Distribución de Marcas Autos"):
            st.session_state.show_pie_chart = not st.session_state.show_pie_chart
        
        if st.session_state.show_pie_chart:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_autos, 'cant_ocompra')
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
            st.markdown('Total marcas: 44' )
            st.markdown("---")

        # GRAFICO 3: evolución costo repuestos por marca autos
        st.subheader('3. Costo de piezas prom. histórico por Marca (autos)')

        # muestro el dataset 
        with st.expander("Ver tabla de datos",):
            st.dataframe(df_rtos_marca_mes[['marca','año_mes','cant_ocompra','cant_piezas_total',
                                    'costo_pieza_prom_hist','var_costo_pieza_prom','monto_total_compras']], hide_index=True,)

        fig17 = create_plot_orion(df_rtos_marca_mes, 'costo_pieza_prom_hist', 'marca', None, 'Costo Promedio')
        st.plotly_chart(fig17, use_container_width=True)
        st.markdown("---")

        # muestro el grafico torta MARCA CAMIONES
        if st.button("Mostrar/Ocultar Distribución de Marcas Camiones"):
            st.session_state.show_pie_chart_2 = not st.session_state.show_pie_chart_2
        
        if st.session_state.show_pie_chart_2:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_camiones, 'cant_ocompra')
            st.plotly_chart(fig_pie, use_container_width=True)
            st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
            st.text('Total marcas: 26')
            st.markdown("---")

        # GRAFICO 4: evolución costo repuestos por marca camiones
        st.subheader('4. Costo de piezas prom. histórico por Marca (camiones)')
        with st.expander("Ver tabla de datos",):
            st.dataframe(df_rtos_marca_mes_cam[['marca','año_mes','cant_ocompra','cant_piezas_total',
                                    'costo_pieza_prom_hist','var_costo_pieza_prom','monto_total_compras']], hide_index=True,)

        fig20 = create_plot_orion(df_rtos_marca_mes_cam, 'costo_pieza_prom_hist', 'marca', None, 'Costo Promedio')
        st.plotly_chart(fig20, use_container_width=True)
        st.markdown("---")      

        # GRAFICO 5: evolución costo mano de obra por tva y tipo de mano de obra
        st.subheader('5. Costo de mano de obra prom. histórico por Tipo de M.O y TVA')

        # muestro el dataset
        with st.expander("Ver tabla de datos (resumen)",):
            st.dataframe(df_cm_mo_resumen[['tva','año','cant_perit',
                                    'cm_hs_chapa','var_cm_hs_chapa','cm_hs_elect', 'var_cm_hs_elect',
                                    'cm_hs_mec', 'var_cm_hs_mec', 'cm_hs_pint', 'var_cm_hs_pint', 'cm_hs_total', 'var_cm_hs_total']], hide_index=True,)

        
        df_cm_mo_hist = df_cm_mo_hist[
            (df_cm_mo_hist['tva'] == 'moto') & (df_cm_mo_hist['tipo_costo'] != 'cm_hs_elect') |
            (df_cm_mo_hist['tva'] != 'moto')
        ]
        fig11 = create_plot_orion(df_cm_mo_hist, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio', 45)
        st.plotly_chart(fig11, use_container_width=True)
        st.markdown("---")

        # gráfico 6: evolución costo mano de obra cleas si vs cleas no
        st.subheader('6. Comparativa variación M.O - CLEAS SI vs CLEAS NO')
        # muestro el dataset
        with st.expander("Ver tabla de datos (resumen)",):
            st.dataframe(df_cm_mo_cleas_resumen[['tva','año','cant_perit',
                                    'cm_hs_chapa','var_cm_hs_chapa','cm_hs_elect', 'var_cm_hs_elect',
                                    'cm_hs_mec', 'var_cm_hs_mec', 'cm_hs_pint', 'var_cm_hs_pint', 'cm_hs_total', 'var_cm_hs_total']], hide_index=True,)
                
        # quito camion_cleas_si del df resumen por poca cantidad de datos
        df_cm_mo_hist_cleas = df_cm_mo_hist_cleas[df_cm_mo_hist_cleas['tva'] != 'camion_cleas_si']
        fig14 = create_plot_orion(df_cm_mo_hist_cleas, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio', 45)
        st.plotly_chart(fig14, use_container_width=True)
        
    # ----- GRAFICOS AJUSTADOS POR IPC --------------------------------------------------
    elif st.session_state['selected_variation_type'] == "IPC":

        # gráfico 1: evolución costo repuestos por tva IPC
        st.subheader('1. Evolución del costo prom. por TVA - Ajust. por IPC')

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_rep_tv[['tva','año_mes','cant_ocompra','cant_piezas_total','var_cant_piezas',
                                    'cant_piezas_prom','monto_total_compras','ipc','monto_ipc','costo_prom_ipc','var_costo_prom_ipc']], hide_index=True,)

        fig7 = create_plot_orion(df_rep_tv, 'costo_prom_ipc', 'tva', None, 'Costo Promedio Ajust. por IPC')
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("---")
    
        # gráfico 2: evolución costo repuestos por tipo repuesto IPC
        st.subheader('2. Evolución del costo prom. por Tipo Repuesto - Ajust. por IPC')

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_tipo_rep[['año', 'año_mes', 'cant_ocompra', 'cant_piezas_total',
                        'cant_piezas_prom', 'ipc', 'monto_ipc', 'costo_prom_ipc',
                        'var_costo_prom_ipc', 'tipo_repuesto']], hide_index=True)

        fig8 = create_plot_orion(df_tipo_rep, 'costo_prom_ipc', 'tipo_repuesto', None,'Costo Promedio ajust. por IPC')
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("---")

        # muestro grafico torta MARCA AUTOS
        if st.button("Mostrar/Ocultar Distribución de Marcas Autos"):
            st.session_state.show_pie_chart = not st.session_state.show_pie_chart
        
        if st.session_state.show_pie_chart:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_autos, 'cant_ocompra')
            st.plotly_chart(fig_pie, use_container_width=True)
            st.text('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
            st.text('Total marcas: 44' )
            st.markdown("---")

        # gráfico 3: evolución costo repuestos por marca autos IPC
        st.subheader('3. Costo de piezas prom. por Marca (autos) - Ajust. por IPC')

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            st.dataframe(df_rtos_marca_mes[['marca','año_mes','cant_ocompra','cant_piezas_total',
                                    'costo_prom_ipc','var_costo_prom_ipc','monto_ipc']], hide_index=True,)

        fig18 = create_plot_orion(df_rtos_marca_mes, 'costo_prom_ipc', 'marca', None, 'Costo Promedio')
        st.plotly_chart(fig18, use_container_width=True)
        st.markdown("---")

        # muestro grafico torta MARCA CAMIONES
        if st.button("Mostrar/Ocultar Distribución de Marcas Camiones"):
            st.session_state.show_pie_chart_2 = not st.session_state.show_pie_chart_2
        
        if st.session_state.show_pie_chart_2:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_camiones, 'cant_ocompra')
            st.plotly_chart(fig_pie, use_container_width=True)
            st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
            st.text('Total marcas: 26')
            st.markdown("---")

        # gráfico 4: evolución costo repuestos por marca camiones IPC
        st.subheader('4. Costo de piezas prom. por Marca (camiones) - Ajust. por IPC')

        # muestro el dataset 
        with st.expander("Ver tabla de datos",):
            st.dataframe(df_rtos_marca_mes_cam[['marca','año_mes','cant_ocompra','cant_piezas_total',
                                    'costo_prom_ipc','var_costo_prom_ipc','monto_ipc']], hide_index=True,)

        fig21 = create_plot_orion(df_rtos_marca_mes_cam, 'costo_prom_ipc', 'marca', None, 'Costo Promedio')
        st.plotly_chart(fig21, use_container_width=True)
        st.markdown("---")    

        # gráfico 5: evolución costo mano de obra por tva y tipo de mano de obra IPC
        st.subheader('5. Evolución del costo de mano de obra prom. por Tipo de M.O y TVA - Ajust. por IPC')

        # muestro el dataset
        with st.expander("Ver tabla de datos (resumen)",):
            st.dataframe(df_cm_mo_resumen[['tva','año','cant_perit',
                                    'cm_hs_chapa_ipc','var_cm_hs_chapa_ipc','cm_hs_elect_ipc', 'var_cm_hs_elect_ipc',
                                    'cm_hs_mec_ipc', 'var_cm_hs_mec_ipc', 'cm_hs_pint_ipc', 'var_cm_hs_pint_ipc', 'cm_hs_total_ipc', 'var_cm_hs_total_ipc']], hide_index=True,)

        df_cm_mo_ipc = df_cm_mo_ipc[
            (df_cm_mo_ipc['tva'] == 'moto') & (df_cm_mo_ipc['tipo_costo'] != 'cm_hs_elect_ipc') |
            (df_cm_mo_ipc['tva'] != 'moto')
        ]
        fig12 = create_plot_orion(df_cm_mo_ipc, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio ajust. por IPC')
        st.plotly_chart(fig12, use_container_width=True)
        st.markdown("---")

        # gráfico 6: evolución costo mano de obra cleas si vs cleas no IPC
        st.subheader('6. Comparativa variación M.O - CLEAS SI vs CLEAS NO - Ajust. por IPC')

        # muestro el dataset
        with st.expander("Ver tabla de datos (resumen)",):
            st.dataframe(df_cm_mo_cleas_resumen[['tva','año','cant_perit',
                                    'cm_hs_chapa_ipc','var_cm_hs_chapa_ipc','cm_hs_elect_ipc', 'var_cm_hs_elect_ipc',
                                    'cm_hs_mec_ipc', 'var_cm_hs_mec_ipc', 'cm_hs_pint_ipc', 'var_cm_hs_pint_ipc', 'cm_hs_total_ipc', 'var_cm_hs_total_ipc']], hide_index=True,)

        # quito camion_cleas_si del df resumen por poca cantidad de datos
        df_cm_mo_ipc_cleas = df_cm_mo_ipc_cleas[df_cm_mo_ipc_cleas['tva'] != 'camion_cleas_si']
        fig15 = create_plot_orion(df_cm_mo_ipc_cleas, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio ajust. por IPC')
        st.plotly_chart(fig15, use_container_width=True)

    # ----- GRAFICOS EN USD -----
    elif st.session_state['selected_variation_type'] == "USD":

        # gráfico 1: evolución costo repuestos por tva USD    
        st.subheader('1. Evolución del costo prom. por TVA en USD')

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_rep_tv[['tva','año_mes','cant_ocompra','cant_piezas_total','var_cant_piezas',
                                    'cant_piezas_prom','monto_total_compras','usd_blue','monto_usd','costo_prom_usd','var_costo_prom_usd']], hide_index=True,)

        fig9 = create_plot_orion(df_rep_tv, 'costo_prom_usd', 'tva', None, 'Costo Promedio (USD)')
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown("---")

        # gráfico 2: evolución costo repuestos por tipo repuesto USD
        st.subheader('2. Evolución del costo prom. por Tipo Repuesto en USD')

        # muestro el dataset
        with st.expander("Ver tabla de datos"):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_tipo_rep[['año', 'año_mes', 'cant_ocompra', 'cant_piezas_total',
                        'cant_piezas_prom', 'usd_blue', 'monto_usd', 'costo_prom_usd',
                        'var_costo_prom_usd', 'tipo_repuesto']], hide_index=True)

        fig10 = create_plot_orion(df_tipo_rep, 'costo_prom_usd', 'tipo_repuesto', None, 'Costo Promedio (USD)')
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown("---")

        # muestro grafico torta MARCA AUTOS
        if st.button("Mostrar/Ocultar Distribución de Marcas Autos"):
            st.session_state.show_pie_chart = not st.session_state.show_pie_chart
        
        if st.session_state.show_pie_chart:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_autos, 'cant_ocompra')
            st.plotly_chart(fig_pie, use_container_width=True)
            st.text('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
            st.text('Total marcas: 44' )
            st.markdown("---")

        # gráfico 3: evolución costo repuestos por marca autos USD
        st.subheader('3. Costo de piezas prom. histórico por Marca (autos) en USD')

        # muestro el dataset
        with st.expander("Ver tabla de datos"):
            st.dataframe(df_rtos_marca_mes[['marca','año_mes','cant_ocompra','cant_piezas_total', 'usd_blue',
                                    'costo_prom_usd','var_costo_prom_usd','monto_usd']], hide_index=True,)

        fig19 = create_plot_orion(df_rtos_marca_mes, 'costo_prom_usd', 'marca', None, 'Costo Promedio (USD)')
        st.plotly_chart(fig19, use_container_width=True)
        st.markdown("---")

        # muestro grafico torta MARCA CAMIONES
        if st.button("Mostrar/Ocultar Distribución de Marcas Camiones"):
            st.session_state.show_pie_chart_2 = not st.session_state.show_pie_chart_2
        
        if st.session_state.show_pie_chart_2:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_camiones, 'cant_ocompra')
            st.plotly_chart(fig_pie, use_container_width=True)
            st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
            st.text('Total marcas: 26')
            st.markdown("---")     

        # gráfico 4: evolución costo repuestos por marca camiones USD
        st.subheader('4. Costo de piezas prom. por Marca (camiones) en USD')

        # muestro el dataset
        with st.expander("Ver tabla de datos"):
            st.dataframe(df_rtos_marca_mes_cam[['marca','año_mes','cant_ocompra','cant_piezas_total', 'usd_blue',
                                    'costo_prom_usd','var_costo_prom_usd','monto_usd']], hide_index=True,)

        fig22 = create_plot_orion(df_rtos_marca_mes_cam, 'costo_prom_usd', 'marca', None, 'Costo Promedio (USD)')
        st.plotly_chart(fig22, use_container_width=True)
        st.markdown("---") 

        # gráfico 5: evolución costo mano de obra por tva y tipo de mano de obra USD
        st.subheader('5. Evolución del costo de Mano de Obra prom. por Tipo de M.O y TVA en USD')

        # muestro el dataset
        with st.expander("Ver tabla de datos (resumen)",):
            st.dataframe(df_cm_mo_resumen[['tva','año','cant_perit',
                                    'cm_hs_chapa_usd','var_cm_hs_chapa_usd','cm_hs_elect_usd', 'var_cm_hs_elect_usd',
                                    'cm_hs_mec_usd', 'var_cm_hs_mec_usd', 'cm_hs_pint_usd', 'var_cm_hs_pint_usd', 'cm_hs_total_usd', 'var_cm_hs_total_usd']], hide_index=True,)

        df_cm_mo_usd = df_cm_mo_usd[
            (df_cm_mo_usd['tva'] == 'moto') & (df_cm_mo_usd['tipo_costo'] != 'cm_hs_elect_usd') |
            (df_cm_mo_usd['tva'] != 'moto')
        ]
        fig13 = create_plot_orion(df_cm_mo_usd, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio (USD)')
        st.plotly_chart(fig13, use_container_width=True)
        st.markdown("---")

        # gráfico 6: evolución costo mano de obra cleas si vs cleas no USD
        st.subheader('6. Comparativa variación M.O en USD - CLEAS SI vs CLEAS NO')

        # muestro el dataset
        with st.expander("Ver tabla de datos (resumen)",):
            st.dataframe(df_cm_mo_cleas_resumen[['tva','año','cant_perit',
                                    'cm_hs_chapa_usd','var_cm_hs_chapa_usd','cm_hs_elect_usd', 'var_cm_hs_elect_usd',
                                    'cm_hs_mec_usd', 'var_cm_hs_mec_usd', 'cm_hs_pint_usd', 'var_cm_hs_pint_usd', 'cm_hs_total_usd', 'var_cm_hs_total_usd']], hide_index=True,)

        # quito camion_cleas_si del df resumen por poca cantidad de datos
        df_cm_mo_usd_cleas = df_cm_mo_usd_cleas[df_cm_mo_usd_cleas['tva'] != 'camion_cleas_si']
        fig16 = create_plot_orion(df_cm_mo_usd_cleas, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio (USD)')
        st.plotly_chart(fig16, use_container_width=True)


# ---- Análisis por PROVINCIA --------------------------------------------------
elif selected_analysis == "Análisis por Provincia":
    st.title('Análisis Coste Medio por Provincia')     
    st.markdown("---")   
    st.header('Coste Medio de repuestos por provincia')
    st.markdown("#### _Fuente de datos: Orion/Cesvi_")
    

    def create_map_chart(df, selected_coverable, color, selected_fecha):
        df_cm_filtered = df[(df['coverable'] == selected_coverable) &
                            (df['año'] == selected_fecha)]
        # df_cm_filtered = df_cm_filtered[df_cm_filtered['año'] == selected_fecha]
        # if selected_fecha is not None:
        #     df_cm_filtered = df_cm_filtered[df_cm_filtered['fecha_cierre'] == selected_fecha]
            
        # Agregado para el segundo caso (Orion/Cesvi) donde el   campo es 'año'
        # Esta es una mejora para que la misma función sirva para ambos datasets            

        if df_cm_filtered.empty:
            return st.warning("No hay información.")

        min_cost = df_cm_filtered[color].min()
        max_cost = df_cm_filtered[color].max()
        ROUNDING_UNIT = 100000
        min_cost = np.floor(min_cost / ROUNDING_UNIT) * ROUNDING_UNIT

        fig = px.choropleth(
            df_cm_filtered,
            geojson=provincias_geojson,
            locations='provincia',  
            featureidkey="properties.nombre_normalizado", # Usamos la nueva clave normalizada
            color=color,
            color_continuous_scale="oranges",
            range_color=[min_cost, max_cost],
            labels={'coste_medio_prom': 'Coste Medio Promedio'},
            projection="mercator",
            width=1000, 
            height=1000  
        )
        # Ajustes de visualización para el mapa
        fig.update_geos(
            visible=False,
            fitbounds=False,
            showcountries=True,
            showcoastlines=True,
            coastlinecolor="black",
            showland=True,
            landcolor="lightgrey",
            scope="south america",
            projection_scale=1, # Ajusta el zoom del mapa
        )
        return fig

    # ----- Comparativo Orion/Cesvi por provincia --------------------------------------------------
    available_coverables_orion = sorted(df_cm_prov_orion['coverable'].unique().tolist())
    available_fechas = sorted(df_cm_prov_orion['año'].unique().tolist())

    # 2 cols para separar grafico y contenedor de filtros
    col3, col4 = st.columns([1, 4], gap='large') # la segunda col es 4 veces el ancho de la primera 
    
    with col3:  
        with st.container(border=True):
            selected_coverable_map = st.selectbox(
                "Seleccionar coverable:",
                options=available_coverables_orion,   
                index=available_coverables_orion.index('AUT'), 
            )
        with st.container(border=True):
            # contenedor para seleccionar fecha
            selected_fecha = st.selectbox(
                "Seleccionar año:",
                options=available_fechas,  
                index=len(available_fechas)-1 
            )

    df_cm_prov_orion_cov = df_cm_prov_orion[df_cm_prov_orion['coverable'] == selected_coverable_map]

    with col4:
        with st.container(border=True):            
            st.markdown(f"#### Coverable selecccionado: {selected_coverable_map}")
            st.markdown(f"#### Año: {selected_fecha}")
            fig_prov = create_map_chart(df_cm_prov_orion, selected_coverable_map, 'costo_pieza_prom', selected_fecha)
            st.plotly_chart(fig_prov, use_container_width=False)    
    
    st.markdown("#### Tabla comparativa: Coste Medio por provincia - Orion/Cesvi")  
    comparativo_orion_prov_raw = comparativo_orion_prov[(comparativo_orion_prov['coverable'] == selected_coverable_map)]
    st.dataframe(comparativo_orion_prov_raw, use_container_width=True)    
    
    with st.expander("Ver data cruda",):
        st.markdown("#### Data Cruda")
        # Para mostrar los datos crudos filtrados (opcional, ajusta tu lógica de datos)
        df_cm_prov_orion_raw = df_cm_prov_orion[(df_cm_prov_orion['coverable'] == selected_coverable_map) &
                                                    (df_cm_prov_orion['año'] == selected_fecha)]
        st.dataframe(df_cm_prov_orion_raw, use_container_width=True)

    # ----- Comparativo BI La Segunda por provincia --------------------------------------------------
    st.header('Coste Medio siniestral por provincia')
    st.markdown("#### _Fuente de datos: BI La Segunda_")

    available_coverables = sorted(df_cm_agg['coverable'].unique().tolist())
    available_fechas = sorted(df_cm_agg['año'].unique().tolist())

    # 2 cols para separar grafico y contenedor de filtros
    col1, col2 = st.columns([1, 4], gap='large') # la segunda col es 4 veces el ancho de la primera 
    
    with col1:  
        with st.container(border=True):
            selected_coverable_map = st.selectbox(
                "Seleccionar coverable:",
                options=available_coverables,   
                index=available_coverables.index('AUT'), 
            )
        with st.container(border=True):
    # contenedor para seleccionar fecha
            selected_fecha = st.selectbox(
            "Seleccionar año:",
            options=available_fechas,   
            index=len(available_fechas)-1, # por defecto la ultima fecha
            )
            # st.markdown("---")
        # st.markdown(f"**Vehículo Seleccionado:** `{selected_coverable_map}`")

    df_cm_cov_fecha = df_cm_agg[(df_cm_agg['coverable'] == selected_coverable_map)]# &
                                # (df_cm_agg['año'] == selected_fecha)]

    with col2:
        with st.container(border=True):
            # st.subheader(f'Análisis Coste Medio por Provincia - {selected_coverable_map}')
            st.markdown(f"#### Coverable selecccionado: {selected_coverable_map}")
            st.markdown(f"#### Año: {selected_fecha}")
            fig_prov = create_map_chart(df_cm_cov_fecha, selected_coverable_map, 'coste_medio_prom', selected_fecha)
            st.plotly_chart(fig_prov, use_container_width=False)    

    st.markdown("#### Tabla comparativa: Coste Medio siniestral por provincia")  
    comparativo_cm_siniestral_raw = comparativo_cm_siniestral[(comparativo_cm_siniestral['coverable'] == selected_coverable_map)]
    st.dataframe(comparativo_cm_siniestral_raw, use_container_width=True)

    with st.expander("Ver data cruda",):
        st.markdown("#### Data Cruda")
        # Para mostrar los datos crudos filtrados (opcional, ajusta tu lógica de datos)
        df_cm_filtered_raw = df_cm_prov[(df_cm_prov['coverable'] == selected_coverable_map) &
                                                        (df_cm_prov['año'] == selected_fecha)]
        st.dataframe(df_cm_filtered_raw, use_container_width=True)   


# ----- Comparativo Mano de obra --------------------------------------------------
elif selected_analysis == "Comparativo de Mano de Obra (L2/Cesvi)":
    st.title('Comparativo Mano de obra - La Segunda vs CESVI/Sancor/San Cristobal')    
    st.markdown("---")
    
    # sidebar por tipo de variación: histórico, ipc, usd
    with st.sidebar:
        st.markdown("---")
        st.markdown("##### _Seleccionar Tipo de Variación:_")
        selected_variation_type_2 = st.selectbox(
            "Tipo de Variación",
            options=["Histórico", "IPC", "USD"],
            index=0,
            label_visibility='collapsed'
        )
        st.markdown("---")
        # Guardo la selección en session_state para que la funcion pueda usarla
        st.session_state['selected_variation_type_2'] = selected_variation_type_2
    
    def create_plot_mo(df, y_col, color, facet_col, y_label, line_width=2):       
        if df.empty:
            fig = go.Figure().update_layout(
                title_text=f"No hay datos para graficar",
                height=400,
                font=dict(family="Arial", size=10),
                title_font_size=12
            )
            return fig

        fig = px.line(
            df,
            x='anio_mes',
            y=y_col,
            color=color,
            color_discrete_sequence=["#FB0D0D", "lightgreen", "blue", "gray", "magenta", "cyan", "orange", '#2CA02C'],
            facet_col=facet_col,
            # line_group='marca',
            # facet_col='tipo_cristal', # Subplots por tipo de cristal
            #title='', agrego titulo con subheader
            labels={'value': y_label, 'anio_mes': ''}# 'marca': 'Marca', 'tipo_cristal': 'Tipo de Cristal'}
        )

        fig.update_layout(
            legend_title_text='Aseguradora',
            height=400, # Altura del subplot individual
            font=dict(family="Arial", size=15),
            margin=dict(t=50, b=0, l=0, r=0),
        )
        fig.update_traces(line_width=line_width)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        return fig
    
    def create_plot_mo_area(df, y_col, color, facet_col, y_label):

        if df.empty:
            fig = go.Figure().update_layout(
                title_text=f"No hay datos para graficar",
                height=400,
                font=dict(family="Arial", size=10),
                title_font_size=12
            )
            return fig

        # --- CAMBIO CLAVE: Usamos px.bar en lugar de px.line ---
        fig = px.area(
            df,
            x='anio_mes',
            y=y_col,
            color=color,
            color_discrete_sequence=["#FB0D0D", "lightgreen", "blue", "gray", "magenta", "cyan", "orange", '#2CA02C'],
            facet_col=facet_col,
            labels={'value': y_label, 'anio_mes': ''}
        )

        fig.update_layout(
            legend_title_text='Aseguradora',
            height=400, # Altura del subplot individual
            font=dict(family="Arial", size=15),
            margin=dict(t=50, b=0, l=0, r=0),
        )
        
        # Eliminamos fig.update_traces(line_width=...)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        return fig
    
    # ----- GRAFICOS HISTORICOS --------------------------------------------------
    if st.session_state['selected_variation_type_2'] == "Histórico":
        y_cols_hist = ['grupo_cesvi', 'grupo_sls', 'la_segunda', 'san_cristobal', 'sancor']
        
        st.subheader('Evolución monto de Repuestos y Mano de Obra (MO)')
        # mostrar evolutivo MO (Chapa/Pintura)
        if st.button("Mostrar/Ocultar Evolutivo M.O (chapa y pintura)"):
            st.session_state.show_mo = not st.session_state.show_mo
        
        if st.session_state.show_mo:
            st.markdown('#### Evolutivo chapa y pintura')
            fig_mo = create_plot_mo_area(df_chapa_pintura, 'monto_historico', 'aseguradora', 'tipo', 'Monto')
            st.plotly_chart(fig_mo, use_container_width=True)
            with st.expander("Ver tabla de datos",):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_chapa_pintura[['anio_mes','aseguradora','monto_historico','tipo']], hide_index=True, width=1500,)

        fig_1 = create_plot_mo(df_mo_repuestos_final, 'monto_historico', 'aseguradora', 'tipo', 'Monto')
        st.plotly_chart(fig_1, use_container_width=True)

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_mo_repuestos_final[['anio_mes','aseguradora','monto_historico','tipo']], hide_index=True, width=1000,)


        st.subheader('Evolución monto de reparaciones (Repuestos + MO)')

        fig_3 = create_plot_mo(df_tot_reparacion, y_cols_hist, None, None, 'Monto MO')
        st.plotly_chart(fig_3, use_container_width=True)
        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_tot_reparacion, hide_index=True,)

        st.subheader('Evolución costo hora de Mano de Obra')
        fig_5 = create_plot_mo(df_costo_hora, y_cols_hist, None, None, 'Costo hora')
        st.plotly_chart(fig_5, use_container_width=True)
        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_costo_hora, hide_index=True,)

        st.subheader('Peritaciones', divider='grey')

        st.subheader('- Evolución cantidad de Peritaciones')
        fig_4 = create_plot_mo(df_peritaciones, y_cols_hist, None, None, 'Cantidad de Peritaciones')
        st.plotly_chart(fig_4, use_container_width=True)

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_peritaciones[['anio_mes', 'grupo_cesvi', 'grupo_sls', 'la_segunda', 'san_cristobal', 'sancor']], hide_index=True, width=1000,)

        st.subheader('- % Variación mensual de cantidad de Peritaciones')
        y_var=['var_%_grupo_cesvi', 'var_%_grupo_sls', 'var_%_la_segunda', 'var_%_san_cristobal', 'var_%_sancor']
        fig_5 = create_plot_mo(df_peritaciones, y_var, None, None, '% variación')
        st.plotly_chart(fig_5, use_container_width=True)

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_peritaciones[['anio_mes', 'part_grupo_sls_vs_cesvi', 'part_la_segunda_vs_cesvi', 'part_san_cristobal_vs_cesvi', 'part_sancor_vs_cesvi']], 
                         hide_index=True, width=1000,)
            
        st.subheader('- % Participacion respecto a Grupo Cesvi')
        y_cols_part=['part_grupo_sls_vs_cesvi', 'part_la_segunda_vs_cesvi', 'part_san_cristobal_vs_cesvi', 'part_sancor_vs_cesvi']
        fig_6 = create_plot_mo(df_peritaciones, y_cols_part, None, None, '% participacion')
        st.plotly_chart(fig_6, use_container_width=True)

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_peritaciones[['anio_mes', 'var_%_grupo_cesvi', 'var_%_grupo_sls', 'var_%_la_segunda', 'var_%_san_cristobal', 'var_%_sancor']], 
                         hide_index=True, width=1000,)

# ----- GRAFICOS IPC --------------------------------------------------
    if st.session_state['selected_variation_type_2'] == "IPC":
        y_cols_ipc = ['grupo_cesvi_ipc', 'grupo_sls_ipc', 'la_segunda_ipc', 'san_cristobal_ipc', 'sancor_ipc']
        
        st.subheader('Evolución monto de Repuestos y Mano de Obra (MO) - ajust. por IPC')
        # mostrar evolutivo MO (Chapa/Pintura)
        if st.button("Mostrar/Ocultar Evolutivo M.O (chapa y pintura)"):
            st.session_state.show_mo = not st.session_state.show_mo
        
        if st.session_state.show_mo:
            st.markdown('#### Evolutivo chapa y pintura IPC')
            fig_mo = create_plot_mo_area(df_chapa_pintura, 'monto_ipc', 'aseguradora', 'tipo', 'Monto')
            st.plotly_chart(fig_mo, use_container_width=True)
            with st.expander("Ver tabla de datos",):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_chapa_pintura[['anio_mes','aseguradora','monto_ipc','tipo']], hide_index=True, width=1500,)

        fig_1 = create_plot_mo(df_mo_repuestos_final, 'monto_ipc', 'aseguradora', 'tipo', 'Monto')
        st.plotly_chart(fig_1, use_container_width=True)

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_mo_repuestos_final[['anio_mes','aseguradora','monto_ipc','tipo']], hide_index=True, width=1500,)


        st.subheader('Evolución monto de reparaciones (Repuestos + MO) - ajust. por IPC')

        fig_3 = create_plot_mo(df_tot_reparacion, y_cols_ipc, None, None, 'Monto MO')
        st.plotly_chart(fig_3, use_container_width=True)
        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_tot_reparacion[['anio_mes','ipc','grupo_cesvi_ipc', 'grupo_sls_ipc', 'la_segunda_ipc', 'san_cristobal_ipc', 'sancor_ipc']], 
                         hide_index=True, width=1000,)

        st.subheader('Evolución Costo Hora de Mano de Obra - ajust. por IPC')
        fig_5 = create_plot_mo(df_costo_hora, y_cols_ipc, None, None, 'Costo hora')
        st.plotly_chart(fig_5, use_container_width=True)

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_costo_hora[['anio_mes','ipc','grupo_cesvi_ipc', 'grupo_sls_ipc', 'la_segunda_ipc', 'san_cristobal_ipc', 'sancor_ipc']], hide_index=True,)

# ----- GRAFICOS USD --------------------------------------------------
    if st.session_state['selected_variation_type_2'] == "USD":
        y_cols_usd = ['grupo_cesvi_usd', 'grupo_sls_usd', 'la_segunda_usd', 'san_cristobal_usd', 'sancor_usd']
        
        st.subheader('Evolución monto de Repuestos y Mano de Obra (MO) - en USD')
        # mostrar evolutivo MO (Chapa/Pintura)
        if st.button("Mostrar/Ocultar Evolutivo M.O (chapa y pintura)"):
            st.session_state.show_mo = not st.session_state.show_mo
        
        if st.session_state.show_mo:
            st.markdown('#### Evolutivo chapa y pintura en USD')
            fig_mo = create_plot_mo_area(df_chapa_pintura, 'monto_usd', 'aseguradora', 'tipo', 'Monto')
            st.plotly_chart(fig_mo, use_container_width=True)
            with st.expander("Ver tabla de datos",):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_chapa_pintura[['anio_mes','aseguradora','monto_usd','tipo']], hide_index=True, width=1500,)

        fig_1 = create_plot_mo(df_mo_repuestos_final, 'monto_usd', 'aseguradora', 'tipo', 'Monto')
        st.plotly_chart(fig_1, use_container_width=True)

        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_mo_repuestos_final[['anio_mes','aseguradora','monto_usd','tipo']], hide_index=True, width=1500,)


        st.subheader('Evolución monto de reparaciones (Repuestos + MO) - en USD')

        fig_3 = create_plot_mo(df_tot_reparacion, y_cols_usd, None, None, 'Monto MO')
        st.plotly_chart(fig_3, use_container_width=True)
        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_tot_reparacion[['anio_mes','usd_blue','grupo_cesvi_usd', 'grupo_sls_usd', 'la_segunda_usd', 'san_cristobal_usd', 'sancor_usd']], 
                         hide_index=True, width=1000,)

        st.subheader('Evolución Costo Hora de Mano de Obra - en USD')
        fig_5 = create_plot_mo(df_costo_hora, y_cols_usd, None, None, 'Costo hora')
        st.plotly_chart(fig_5, use_container_width=True)
        # muestro el dataset
        with st.expander("Ver tabla de datos",):
            # st.subheader("Tabla de Datos de Ejemplo")
            st.dataframe(df_costo_hora[['anio_mes','usd_blue','grupo_cesvi_usd', 'grupo_sls_usd', 'la_segunda_usd', 'san_cristobal_usd', 'sancor_usd']], 
                         hide_index=True, width=1000,)

