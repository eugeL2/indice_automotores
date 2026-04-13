import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# ----- Cargo las bases de datos --------------------------------------------------
try:
    # df PKT (cristales)
    df_cristal = pd.read_csv('data/base_pkt_nov.csv')
    df_pagos_cristal = pd.read_csv('data/pagos_pkt_ok.csv')

    # dfs de repuestos orion/cesvi
    df_tipo_rep = pd.read_csv('data/df_tipo_rep_feb.csv')
    df_rep_tv = pd.read_csv('data/df_rep_tv_feb.csv')
    # df para graf torta
    df_rep_torta = pd.read_csv('data/todos_los_rep_orion_auto_ok.csv')

    # dfs de mano de obra orion/cesvi
    df_cm_mo = pd.read_csv('data/df_cm_mo_feb.csv')

    # dfs mano de obra cleas si/cleas no
    df_cm_mo_cleas = pd.read_csv('data/df_cm_mo_cleas_feb.csv')

    # dfs marcas
    df_marcas_autos = pd.read_csv('data/evol_todas_marcas_autos.csv')
    df_rtos_marca_mes = pd.read_csv('data/df_rtos_marca_mes_feb.csv')
    df_marcas_camiones = pd.read_csv('data/camion_marcas.csv')
    df_rtos_marca_mes_cam = pd.read_csv('data/df_rtos_marca_mes_cam_feb.csv')
    # df_marcas_cartera = pd.read_csv(r'data\todas_las_marcas_bi.csv')

    # dfs var x prov
    df_cm_prov_orion = pd.read_csv('data/base_cm_x_rep_prov_orion.csv')
    df_cm_prov = pd.read_parquet('data/base_cm_prov_actual.parquet')
    with open('data/prov.geojson', 'r', encoding='utf-8') as f:
        provincias_geojson = json.load(f)
    comparativo_orion_prov = pd.read_parquet('data/comparativo_orion_prov.parquet')
    comparativo_cm_siniestral = pd.read_parquet('data/comparativo_bi_cm_prov.parquet')

    # dfs comparativo mano de obra
    df_chapa_pintura = pd.read_csv('data/df_chapa_pintura_100426.csv')
    df_mo_repuestos_final = pd.read_csv('data/df_mo_repuestos_final_100426.csv')
    df_peritaciones = pd.read_csv('data/df_peritaciones_100426.csv')
    df_costo_hora = pd.read_csv('data/df_costo_hora_100426.csv')
    df_tot_reparacion = pd.read_csv('data/df_tot_reparacion_100426.csv')

    # df pagos ruedas
    df_pagos_ruedas = pd.read_csv('data/pagos_ruedas_ok.csv')
    # df pagos materiales
    df_pagos_materiales = pd.read_csv('data/pagos_dano_mat_ok.csv', keep_default_na=False, na_values=[])
    # df pagos cascos
    df_pagos_cascos = pd.read_csv('data/pagos_cascos_ok.csv')

    # tablas aux
    tabla_marcas_año = pd.read_parquet('data/tabla_20242025_medidas.parquet')
    tabla_marcas_head_8 = pd.read_parquet('data/tabla_marcas_head.parquet')
    tabla_marcas_head_20 = pd.read_parquet('data/tabla_marcas_head_20.parquet')
    df_graf_cartera = pd.read_parquet('data/df_grafico_cartera.parquet')

    # sa vs rep - info autos
    df_sa_rep = pd.read_csv('data/df_sa_rep.csv')
    # sa vs rep - cartera L2
    df_sa_rep_cartera = pd.read_parquet('data/df_sa_rep_cartera.parquet') # sa cartera todos los planes y plan 30 / costo prom repuestos orion
    df_sa_rep_cartera_marcas = pd.read_parquet('data/df_sa_rep_cartera_marcas.parquet')

    # ruedas - grant
    df_ruedas_grant = pd.read_parquet('data/costo_medio_ruedas_grant.parquet')
    df_neum_grant = pd.read_parquet('data/costo_prom_neumatico.parquet')

    # # INDICE
    # indice_hist = pd.read_excel('data/bases_indice.xlsx', sheet_name='indice_hist')
    # indice_ipc = pd.read_excel('data/bases_indice.xlsx', sheet_name='indice_ipc')
    # indice_usd = pd.read_excel('data/bases_indice.xlsx', sheet_name='indice_usd')

except FileNotFoundError as e:
    st.error(f"Error: No se encuentra el archivo CSV. \nDetalles: {e}")
    # La app se detiene si no encuentra los archivos
    st.stop()

# ==========================================================================
# ---- FORMATEO de datos ---------------------------------------------------
# ==========================================================================

# ---- Formateo base cristales --------------------------------------------------
if 'fecha' in df_cristal.columns:
    df_cristal['fecha'] = pd.to_datetime(df_cristal['fecha'])
if 'año_mes' in df_cristal.columns:
    df_cristal['año_mes'] = pd.to_datetime(df_cristal['año_mes'], format='%Y-%m-%d')
if 'cristal' in df_cristal.columns:
    df_cristal['cristal'] = df_cristal['cristal'].astype(str).str.replace('_', ' ').str.title()
df_cristal['marca'] = df_cristal['marca'].astype(str)
if 'zona' in df_cristal.columns:
    df_cristal['zona'] = df_cristal['zona'].astype(str)
if 'tipo_repuesto' in df_cristal.columns:
    df_cristal['tipo_repuesto'] = df_cristal['tipo_repuesto'].astype(str).str.replace('_', ' ').str.title()

# ==========================================================================

# ---- Variables de sesión para mostrar/ocultar gráficos --------------------------------------------------
if 'show_pie_chart' not in st.session_state:
    st.session_state.show_pie_chart = False

if 'show_pie_chart_2' not in st.session_state:
    st.session_state.show_pie_chart_2 = False

if 'show_pie_chart_4' not in st.session_state:
    st.session_state.show_pie_chart_4 = False

if 'show_mo' not in st.session_state:
    st.session_state.show_mo = False

col_izq, col_center, col_der = st.columns([2, 5, 2])

with col_center:
    """
    # :chart_with_upwards_trend: Proyecto: Indice Automotores
    """
    # st.markdown('---')
    st.subheader('', divider='grey')
    st.markdown('')

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

opcion_0 = "Resumen cartera La Segunda"
opcion_1 = "Evolutivo lista de precios Pilkington"
opcion_3 = "Evolutivo precios ORION/CESVI"
opcion_4 = "Análisis Coste Medio por Provincia"
opcion_5 = "Comparativo de Mano de Obra (L2 vs Cesvi vs Colegas)"
opcion_pagos_pkt = "Evolutivo pagos Cristales"
opcion_6 = "Evolutivo pagos Robo de Ruedas"
opcion_7 = "Evolutivo pagos Daños Materiales"
opcion_8 = "Evolutivo pagos Cascos"
opcion_9 = "Variación SA vs. Precio de repuestos"
opcion_10 = "Costo Medio ruedas - GRANT"
opcion_11 = "INDICE"

OPTIONS = [opcion_0, opcion_1, opcion_3, opcion_4, opcion_5, 
           opcion_pagos_pkt, opcion_6, opcion_7, opcion_8, opcion_9, opcion_10, opcion_11]

# ==========================================================================
# PANTALLA INICIO DE LA APP
# ==========================================================================
if 'analysis_selected' not in st.session_state:
    st.session_state.analysis_selected = None
if 'show_initial_selector' not in st.session_state:
    # show_initial_selector es un flag para decidir dónde poner el selector.
    st.session_state.show_initial_selector = True

def handle_initial_selection():
    # Esta función se llama cuando el usuario elige una opción por primera vez.
    if st.session_state.initial_selector_value is not None:
        st.session_state.analysis_selected = st.session_state.initial_selector_value
        st.session_state.show_initial_selector = False

if st.session_state.show_initial_selector:   
   
    # El índice se establece como None para que no haya selección por defecto visible
    with col_center:
        st.markdown("### Seleccionar un análisis para comenzar:")
        initial_selection = st.selectbox(
            'Seleccionar Análisis:',
            options=OPTIONS,
            index=None,
            label_visibility ='collapsed',
            key="initial_selector_value", # Clave para acceder al valor en el callback
            placeholder= "Selecciona una opción...", 
            on_change=handle_initial_selection  
        )
    
# --- Selector en el Sidebar y Contenido del Análisis ---
else:
    
    # Selector de Análisis en el Sidebar
    st.sidebar.markdown("## 🔄 Cambiar Análisis")
    selected_analysis_sidebar = st.sidebar.selectbox(
        'Análisis Seleccionado:',
        options=OPTIONS,
        index=OPTIONS.index(st.session_state.analysis_selected), # Mantiene la última selección
        label_visibility ='collapsed',
        key="sidebar_selector_value",
        on_change=lambda: st.session_state.update(analysis_selected=st.session_state.sidebar_selector_value)
    )
    
    # 2. Renderizado del Contenido del Análisis
    current_analysis = st.session_state.analysis_selected



# ==========================================================================
# ---- Análisis PILKINGTON -------------------------------------------------
# ==========================================================================
    if current_analysis == opcion_1:
        st.markdown("## Variación de precios de Cristales y Mano de obra por Marca y Zona")
        st.markdown("#### _Fuente de datos: Listas de precios de Pilkington_ \n Fecha de actualización: **01/11/2025**") 
        st.markdown("---")

        # Dropdown de Zona (barra lateral)
        available_zones = sorted(df_cristal['zona'].unique().tolist())
        available_marcas = sorted(df_cristal['marca'].unique().tolist())
        # available_cristales = sorted(df_cristal['cristal'].unique().tolist())

        DEFAULT_MARCAS = ["VOLKSWAGEN", "CHEVROLET", "FORD",  "TOYOTA", "FIAT", "PEUGEOT", "RENAULT"]

        with st.sidebar:
            st.markdown("---")

            st.markdown("##### _Seleccionar Zona:_") 
            selected_zone = st.selectbox(
                "Zona",
                options=available_zones,
                index=0, # primera zona por defecto,
                label_visibility ='collapsed'
            )

            st.markdown("---")
            
            st.markdown("##### _Seleccionar Marcas:_")
            selected_marcas = st.multiselect(
                "Marcas",
                options=available_marcas,
                default=[m for m in DEFAULT_MARCAS if m in available_marcas],
                label_visibility='collapsed',
            )
            st.markdown("---")

        def create_plot_pkt(df_source, y_col, y_label, title, x_tickangle=45):   

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
                labels={'fecha': '', y_col: y_label, 'marca': 'Marca', 'cristal': 'Tipo de Cristal'},
                title = title
            )

            fig.update_traces(
                    mode="lines+markers",
                    marker=dict(size=4),
                    hovertemplate="<b>%{fullData.name}:</b> $ %{y:,.0f}<extra></extra>" # Muestra solo el valor con separador de miles
                )

            # Ajustes del gráfico
            fig.update_layout(
                height=600, # Altura del subplot individual
                legend_title_text='Marca',
                font=dict(family="Arial", size=15),
                # margin=dict(t=100, b=0, l=0, r=0),
                hovermode="x unified",
                legend=dict(
                    orientation="h",      
                    yanchor="top",       
                    y=-0.20,              
                    xanchor="center",     
                    x=0.5,                
                    title_text=""         
                ),
                # Ajustamos el margen inferior (b) para que la leyenda no se corte
                margin=dict(t=80, b=120, l=50, r=20),
                title=dict(
                    font=dict(
                        size=18,  # <-- Aumenta este valor para un título más grande (ej: 24, 28, etc.)
                        family="Arial",
                        # color="black" # Opcional: puedes cambiar el color también
                    ),
                )
            )
            fig.for_each_xaxis(
            lambda xaxis: xaxis.update(
                tickangle=x_tickangle, # Aplicar el ángulo deseado
                showticklabels=True     # Asegurarse de que las etiquetas sean visibles (si fuera necesario)
                )
            )
            # Ajustar el título de las facetas para que sean más legibles
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            
            return fig

        # ==========================================================================
        # ----- GRAFICOS HISTORICOS, IPC y USD -------------------------------------

        if not selected_marcas:
            st.warning(":warning: Seleccionar una marca para ver la información.")
            st.stop()

        else:

            with st.container(border=True):
                # st.subheader('1. Precios de Material históricos (Sin IVA)')
                fig1 = create_plot_pkt(df_cristal, 'precio', 'Precio Sin IVA', '1. Precios de Material históricos (Sin IVA)')
                
                ipc_data = df_cristal[['fecha', 'var_ipc']].drop_duplicates().sort_values('fecha')

                fig1_ipc_ = create_plot_pkt(df_cristal, 'var_precio_prom', 'Variación (base 1)', '1b. Variación Precios de Material históricos vs IPC')

                NUM_COLUMNS = 5
                for col_num in range (1, NUM_COLUMNS + 1):

                    mostrar_leyenda = (col_num==1)
            
                    fig1_ipc_.add_trace(go.Scatter(
                        x=ipc_data['fecha'],
                        y=ipc_data['var_ipc'],
                        name='IPC', 
                        mode='lines',
                        line=dict(color='white', dash='dot'), # Cambié a negro para asegurar visibilidad
                        showlegend=mostrar_leyenda,     
                    ),
                    row=1, col=col_num)
        
                fig1_ipc_.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")

                tab1, tab2 = st.tabs(["Evolutivo precios", "Variación vs IPC"])
                with tab1:
                    st.plotly_chart(fig1, use_container_width=True)
                with tab2:
                    st.plotly_chart(fig1_ipc_, use_container_width=True)
                
            st.markdown('')
    # ==========================================================================

            with st.container(border=True):
                # st.subheader('2. Costo de Instalación histórico (Sin IVA)')
                fig2 = create_plot_pkt(df_cristal, 'instalacion', 'Costo de Instalación','2. Costo de Instalación histórico (Sin IVA)')
                
                fig2_ipc_ = create_plot_pkt(df_cristal, 'var_instal_prom', 'Variación (base 1)', '2b. Variación costo Instalación históricos vs IPC')
                
                NUM_COLUMNS = 5
            
                for col_num in range (1, NUM_COLUMNS + 1):

                    mostrar_leyenda = (col_num==1)
            
                    fig2_ipc_.add_trace(go.Scatter(
                        x=ipc_data['fecha'],
                        y=ipc_data['var_ipc'],
                        name='IPC', 
                        mode='lines',
                        line=dict(color='white', dash='dot'), # Cambié a negro para asegurar visibilidad
                        showlegend=mostrar_leyenda,     
                    ),
                    row=1, col=col_num)
                fig2_ipc_.update_layout(legend_title_text='Variación')

                tab1, tab2 = st.tabs(["Evolutivo precios", "Variación vs IPC"])
                with tab1:
                    st.plotly_chart(fig2, use_container_width=True)
                with tab2:
                    st.plotly_chart(fig2_ipc_, use_container_width=True)
            
            st.markdown('')
    # ==========================================================================

            with st.container(border=True):
                # st.subheader('3. Precios de Material (Ajustados por IPC)')
                fig3 = create_plot_pkt(df_cristal, 'precio_ipc', 'Precio (IPC)','3. Precios de Material (Ajustados por IPC)')
                st.plotly_chart(fig3, use_container_width=True)
    # ==========================================================================

            with st.container(border=True):
                # st.subheader('4. Costo de Instalación (Ajustados por IPC)')
                fig4 = create_plot_pkt(df_cristal, 'instalacion_ipc', 'Costo de Instalación (IPC)','4. Costo de Instalación (Ajustados por IPC)')
                st.plotly_chart(fig4, use_container_width=True)
    # ==========================================================================

            with st.container(border=True):
                # st.subheader('5. Precios de Material (USD)')
                fig5 = create_plot_pkt(df_cristal, 'precio_usd', 'Precio (USD)','5. Precios de Material (USD)')
                st.plotly_chart(fig5, use_container_width=True)
    # ==========================================================================

            with st.container(border=True):
                # st.subheader('6. Costo de Instalación (USD)')
                fig6 = create_plot_pkt(df_cristal, 'instalacion_usd', 'Costo de Instalación (USD)','6. Costo de Instalación (USD)')
                st.plotly_chart(fig6, use_container_width=True)     

            ""
            ""
            st.markdown("#### Data Cruda")

            df_filtered_raw = df_cristal[
                (df_cristal['zona'] == selected_zone) &
                (df_cristal['marca'].isin(selected_marcas))
            ].copy()
            df_filtered_raw['fecha'] = df_filtered_raw['fecha'].dt.strftime('%Y-%m-%d')
            st.dataframe(df_filtered_raw[['fecha', 'marca', 'zona', 'cristal', 'precio', 'precio_ipc',
                'precio_usd', 'instalacion', 'instalacion_ipc', 'instalacion_usd', 'ipc_empalme_ipim',
                'ipc',  'var_ipc_%', 'var_precio_prom_%', 'var_instal_prom_%']], use_container_width=True)

# ==========================================================================
# ---- Análisis ORION/CESVI ------------------------------------------------
# ==========================================================================
    elif current_analysis == opcion_3:
        st.title('Variación de Precios de Repuestos y Mano de obra')
        st.markdown("#### _Fuente de datos: Orion/Cesvi_ \nFecha actualización: **diciembre 2025**")
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
                #title='', agrego titulo con subheader
                labels={'año_mes': '', y_col: y_label,}# 'marca': 'Marca', 'tipo_cristal': 'Tipo de Cristal'}
            )

            fig.update_traces(
                    mode="lines+markers",
                    marker=dict(size=4),
                    hovertemplate="<b>%{fullData.name}:</b> $ %{y:,.0f}<extra></extra>" # Muestra solo el valor con separador de miles
                )

            fig.update_layout(
                hovermode="x unified",
                height=400, # Altura del subplot individual
                font=dict(family="Arial", size=15),
                margin=dict(t=50, b=0, l=0, r=0),
            )

            fig.for_each_xaxis(
            lambda xaxis: xaxis.update(
                tickangle=x_tickangle, # Aplicar el ángulo deseado
                showticklabels=True     # Asegurarse de que las etiquetas sean visibles (si fuera necesario)
                )
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            
            return fig
        
        
# ----- GRAFICOS HISTORICOS --------------------------------------------------
        if st.session_state['selected_variation_type'] == "Histórico":
            
            # GRAFICO 1: evolución costo repuestos por tva
            st.subheader('1. Costo de piezas prom. histórico por TVA')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_rep_tv[['tva','año_mes','cant_ocompra','cant_piezas_total','var_cant_piezas',
                                        'cant_piezas_prom','costo_pieza_prom_hist','var_costo_prom_%','ipc','ipc_empalme_ipim','var_ipc_%','monto_total_compras']], 
                                        hide_index=True,)

            fig5 = create_plot_orion(df_rep_tv, 'costo_pieza_prom_hist', 'tva', None,'Costo Promedio')

        
            # 2. Preparar los datos del IPC (evitando duplicados por mes)
            df_ipc_data = df_rep_tv[['año_mes', 'var_ipc']].drop_duplicates().sort_values('año_mes')

            fig5_ipc = create_plot_orion(df_rep_tv, 'var_costo_prom', 'tva', None,'Variación (base 1)')
            # 3. Agregar la línea (trace) del IPC al gráfico existente (fig5)
            fig5_ipc.add_trace(go.Scatter(
                x=df_ipc_data['año_mes'],
                y=df_ipc_data['var_ipc'],
                name='IPC',        # Nombre que aparecerá en la leyenda
                mode='lines',
                line=dict(color='white', dash='dot')
            ))
            fig5_ipc.update_layout(legend_title_text='Variación')
            fig5_ipc.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")

            tab1, tab2 = st.tabs(["Evolutivo CM", "Variación vs IPC"])
            with tab1:
                st.plotly_chart(fig5, use_container_width=True)
            with tab2:
                st.plotly_chart(fig5_ipc, use_container_width=True)

            st.subheader('', divider='grey')
# ==========================================================================

# ----- GRAFICO 2: evolución costo repuestos por tipo repuesto
            def create_pie_chart_repuestos(df, value_col):
                fig = px.pie(
                    df,
                    names='Repuesto',
                    values=value_col,
                    hover_data=[value_col],
                    title='Distribución de montos por Tipo de Repuesto',
                    color_discrete_sequence=px.colors.qualitative.G10,
                    labels={value_col: value_col.title()},
                    hole=0.3
                )
                fig.update_traces(textinfo='percent+label',
                                  insidetextfont=dict(size=14, color='white', family='Arial'))
                fig.update_layout(
                    height=500,
                    font=dict(family="Arial", size=12, color='white'),
                    # title_font_size=16
                    showlegend=False
                )
                return fig
            
            st.subheader('2. Costo de piezas prom. histórico por Tipo Repuesto')
            UMBRAL_PORCENTAJE = 0.03

            try:
                df_rep_torta['Monto Total Compras'] = (
                    df_rep_torta['Monto Total Compras']
                    .astype(str) # Asegurar que es string
                    .str.replace('.', '', regex=False) # Quitar separador de miles
                    .str.replace(',', '.', regex=False) # Reemplazar coma decimal por punto
                )
                df_rep_torta['Monto Total Compras'] = pd.to_numeric(df_rep_torta['Monto Total Compras'], errors='coerce')
                
            except Exception as e:
                st.error(f"Error al limpiar la columna 'Monto Total': {e}") 

            monto_total_general = df_rep_torta['Monto Total Compras'].sum()
            df_rep_torta['% Monto Total'] = df_rep_torta['Monto Total Compras'] / monto_total_general

            df_rep_torta['Repuesto_Agrupado'] = np.where(
                df_rep_torta['% Monto Total'] < UMBRAL_PORCENTAJE,
                'OTRAS',
                df_rep_torta['Pieza']
            )

            df_torta_final = df_rep_torta.groupby('Repuesto_Agrupado').agg(
                {
                    'Monto Total Compras': 'sum',
                    'Cant. O.Compra': 'sum'
                }
            ).reset_index()

            df_torta_final.rename(columns={'Repuesto_Agrupado': 'Repuesto'}, inplace=True)

            # muestro grafico torta MARCA AUTOS
            if st.button("Mostrar/Ocultar Distribución de Repuestos", icon='📊'):
                st.session_state.show_pie_chart_4 = not st.session_state.show_pie_chart_4
            
            if st.session_state.show_pie_chart_4:   
                fig_pie_rep = create_pie_chart_repuestos(df_torta_final, 'Monto Total Compras')
                st.plotly_chart(fig_pie_rep, use_container_width=True)
                st.markdown(f"Total ord. compra (ene23-oct25): **{df_torta_final['Cant. O.Compra'].sum():,.0f}**")
                # Formateo del monto total (suponiendo que es dinero)
                st.markdown(f"Monto Total de Órdenes (ene23-oct25): **${df_torta_final['Monto Total Compras'].sum():,.0f}**") 
                st.markdown(f"Total repuestos únicos: **{df_rep_torta.Pieza.nunique()}**")
                df_rep_torta['% Monto Total'] = df_rep_torta['% Monto Total'] * 100
                df_rep_torta['% Monto Total'] = df_rep_torta['% Monto Total'].round(2).astype(str) + ' %'

                st.dataframe(df_rep_torta.drop(columns=['Repuesto_Agrupado', 'Cant. Piezas (Prom)', 'Cant. Piezas Total'],
                                                axis=1).sort_values(by='Monto Total Compras', ascending=False),
                                                hide_index=True)

                st.markdown("---")

            # muestro distribución MARCA AUTOS
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                st.dataframe(df_tipo_rep[['tipo_repuesto','año_mes','cant_ocompra','cant_piezas_total','var_cant_piezas',
                                        'cant_piezas_prom','costo_pieza_prom_hist','var_costo_prom_%','ipc','ipc_empalme_ipim','var_ipc_%','monto_total_compras']], 
                                        hide_index=True)

            fig6 = create_plot_orion(df_tipo_rep, 'costo_pieza_prom_hist', 'tipo_repuesto', None,'Costo Promedio')

            fig6_ipc = create_plot_orion(df_tipo_rep, 'var_costo_prom', 'tipo_repuesto', None,'Variación (base 1)')
            fig6_ipc.add_trace(go.Scatter(
                x=df_ipc_data['año_mes'],
                y=df_ipc_data['var_ipc'],
                name='IPC',        # Nombre que aparecerá en la leyenda
                mode='lines',
                line=dict(color='white', dash='dot')
            ))
            fig6_ipc.update_layout(legend_title_text='Variación')
            fig6_ipc.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")

            tab1, tab2 = st.tabs(["Evolutivo CM", "Variación vs IPC"])
            with tab1:
                st.plotly_chart(fig6, use_container_width=True)
            with tab2:
                st.plotly_chart(fig6_ipc, use_container_width=True)

            st.subheader('', divider='grey')
            st.markdown('')

# ==============================================================================

            # muestro grafico torta MARCAS AUTOS 
            if st.button("Mostrar/Ocultar Distribución de Marcas Autos",icon='📊'):
                st.session_state.show_pie_chart = not st.session_state.show_pie_chart
            
            if st.session_state.show_pie_chart:
                st.subheader('Distribución de Órdenes de Compra por Marca')
                fig_pie = create_pie_chart(df_marcas_autos, 'cant_ocompra')
                st.plotly_chart(fig_pie, use_container_width=True)
                # st.markdown('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
                st.markdown('Total marcas: 44')
                st.markdown("---")
# ==========================================================================

# ----- GRAFICO 3: evolución costo repuestos por marca autos
            st.subheader('3. Costo de piezas prom. histórico por Marca (autos)')

            # muestro el dataset 
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                st.dataframe(df_rtos_marca_mes[['marca','año_mes','cant_ocompra','cant_piezas_total','var_cant_piezas',
                                        'cant_piezas_prom','costo_pieza_prom_hist','var_costo_prom_%','ipc','ipc_empalme_ipim','var_ipc_%','monto_total_compras']], 
                                        hide_index=True,)

            fig17 = create_plot_orion(df_rtos_marca_mes, 'costo_pieza_prom_hist', 'marca', None, 'Costo Promedio')

            fig7_ipc = create_plot_orion(df_rtos_marca_mes, 'var_costo_prom', 'marca', None,'Variación (base 1)')
            fig7_ipc.add_trace(go.Scatter(
                x=df_ipc_data['año_mes'],
                y=df_ipc_data['var_ipc'],
                name='IPC',        # Nombre que aparecerá en la leyenda
                mode='lines',
                line=dict(color='white', dash='dot')
            ))
            fig7_ipc.update_layout(legend_title_text='Variación')
            fig7_ipc.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")

            tab1, tab2 = st.tabs(["Evolutivo CM", "Variación vs IPC"])
            with tab1:
                st.plotly_chart(fig17, use_container_width=True)
            with tab2:
                st.plotly_chart(fig7_ipc, use_container_width=True)


            st.subheader('', divider='grey')
            st.markdown('')

# ==============================================================================

            # muestro el grafico torta MARCA CAMIONES
            if st.button("Mostrar/Ocultar Distribución de Marcas Camiones", icon='📊'):
                st.session_state.show_pie_chart_2 = not st.session_state.show_pie_chart_2
            
            if st.session_state.show_pie_chart_2:
                st.subheader('Distribución de Órdenes de Compra por Marca')
                fig_pie = create_pie_chart(df_marcas_camiones, 'cant_ocompra')
                st.plotly_chart(fig_pie, use_container_width=True)
                # st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
                st.text('Total marcas: 26')
                st.markdown("---")
# ==========================================================================

# ----- GRAFICO 4: evolución costo repuestos por marca camiones -
            st.subheader('4. Costo de piezas prom. histórico por Marca (camiones)')
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                st.dataframe(df_rtos_marca_mes_cam[['marca','año_mes','cant_ocompra','cant_piezas_total','var_cant_piezas',
                                        'cant_piezas_prom','costo_pieza_prom_hist','var_costo_prom_%','ipc','ipc_empalme_ipim','var_ipc_%','monto_total_compras']],
                                        hide_index=True,)

            fig20 = create_plot_orion(df_rtos_marca_mes_cam, 'costo_pieza_prom_hist', 'marca', None, 'Costo Promedio')

            fig20_ipc = create_plot_orion(df_rtos_marca_mes_cam, 'var_costo_prom', 'marca', None,'Variación (base 1)')
            fig20_ipc.add_trace(go.Scatter(
                x=df_ipc_data['año_mes'],
                y=df_ipc_data['var_ipc'],
                name='IPC',        # Nombre que aparecerá en la leyenda
                mode='lines',
                line=dict(color='white', dash='dot')
            ))
            fig20_ipc.update_layout(legend_title_text='Variación')
            fig20_ipc.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")

            tab1, tab2 = st.tabs(["Evolutivo CM", "Variación vs IPC"])
            with tab1:
                st.plotly_chart(fig20, use_container_width=True)
            with tab2:
                st.plotly_chart(fig20_ipc, use_container_width=True)

            st.subheader('', divider='grey')      
# ==========================================================================

            # GRAFICO 5: evolución costo mano de obra por tva y tipo de mano de obra
            st.subheader('5. Costo de mano de obra prom. histórico por Tipo de M.O y TVA')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                st.dataframe(df_cm_mo[['tva','año_mes','tipo_costo','valor_costo_hist',
                                    'var_costo_hist','var_ipc']], hide_index=True, width=1000,)

            df_cm_mo = df_cm_mo[
                (df_cm_mo['tva'] == 'moto') & (df_cm_mo['tipo_costo'] != 'cm_hs_elect') |
                (df_cm_mo['tva'] != 'moto')
            ]
            fig11 = create_plot_orion(df_cm_mo, 'valor_costo_hist', 'tva','tipo_costo', 'Costo Promedio', 45)
            fig11.update_traces(
                    mode="lines+markers",
                    marker=dict(size=2),)

            df_cm_mo_graf = df_cm_mo[
                (df_cm_mo['tva'] == 'auto') & (df_cm_mo['tipo_costo'] != 'cm_hs_mec') |
                (df_cm_mo['tva'] != 'auto')
            ]

            fig11_ipc = create_plot_orion(df_cm_mo_graf, 'var_costo_hist', 'tva', 'tipo_costo', 'Variación (base 1)')
            
            NUM_COLUMNS = 5
        
            for col_num in range (1, NUM_COLUMNS + 1):

                mostrar_leyenda = (col_num==1)
        
                fig11_ipc.add_trace(go.Scatter(
                    x=df_ipc_data['año_mes'],
                    y=df_ipc_data['var_ipc'],
                    name='IPC', 
                    mode='lines',
                    line=dict(color='white', dash='dot'), # Cambié a negro para asegurar visibilidad
                    showlegend=mostrar_leyenda,     
                ),
                row=1, col=col_num)
            
            fig11_ipc.update_layout(legend_title_text='Variación')
            fig11_ipc.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")
                
            df_cm_mo_graf_2 = df_cm_mo[(df_cm_mo['tva']=='auto') & (df_cm_mo['tipo_costo']=='cm_hs_mec')]
            fig11_ipc2 = create_plot_orion(df_cm_mo_graf_2, 'var_costo_hist', 'tva', None, 'Variación (base 1)')
            fig11_ipc2.add_trace(go.Scatter(
                x=df_ipc_data['año_mes'],
                y=df_ipc_data['var_ipc'],
                name='IPC',        # Nombre que aparecerá en la leyenda
                mode='lines',
                line=dict(color='white', dash='dot')
            ))
            fig11_ipc2.update_layout(legend_title_text='Variación')
            fig11_ipc2.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")

            tab1, tab2, tab3 = st.tabs(["Evolutivo CM ",'Variación vs IPC', "Var. cm_hs_mec vs IPC (solo AUT)"])
            with tab1:
                st.plotly_chart(fig11, use_container_width=True)
            with tab2:
                st.plotly_chart(fig11_ipc, use_container_width=True)
            with tab3:
                # Use the native Plotly theme.
                st.plotly_chart(fig11_ipc2, width='content', )

            '''Se grafica aparte variación de cm_hs_mec para Autos (gran salto de CM en junio y sept 2024)'''


            
            st.subheader('', divider='grey')
# ==========================================================================

            # GRAFICO 6: evolución costo mano de obra cleas si vs cleas no
            st.subheader('6. Comparativa variación M.O - CLEAS SI vs CLEAS NO')
            # muestro el dataset
            with st.expander("Ver tabla de datos",icon=":material/query_stats:"):
                st.dataframe(df_cm_mo_cleas[df_cm_mo_cleas['tva'] != 'camion_cleas_si'], 
                            hide_index=True,          
                            column_config={
                                "var_costo": st.column_config.NumberColumn(
                                    "Var. Costo %", 
                                    format="%.2f",
                                ),
                                "var_ipc": st.column_config.NumberColumn(
                                    "Var. CM IPC %", 
                                    format="%.2f",
                                ),
                                "var_costo_usd": st.column_config.NumberColumn(
                                    "Var. CM en USD %", 
                                    format="%.2f",
                                    ),
                                })    
                                                 
            # quito camion_cleas_si del df resumen por poca cantidad de datos
            df_cm_mo_cleas = df_cm_mo_cleas[df_cm_mo_cleas['tva'] != 'camion_cleas_si']
            fig14 = create_plot_orion(df_cm_mo_cleas, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio', 45)
            fig14.update_traces(
                    mode="lines+markers",
                    marker=dict(size=1),)

            df_cm_mo_cleas2 = df_cm_mo_cleas[
                (df_cm_mo_cleas['tva'] == 'auto_cleas_no') & (df_cm_mo_cleas['tipo_costo'] != 'cm_hs_mec') |
                (df_cm_mo_cleas['tva'] != 'auto_cleas_no')
            ]
            fig14_ipc = create_plot_orion(df_cm_mo_cleas2, 'var_costo', 'tva', 'tipo_costo', 'Variación (base 1)')

            NUM_COLUMNS = 5
        
            for col_num in range (1, NUM_COLUMNS + 1):

                mostrar_leyenda = (col_num==1)
        
                fig14_ipc.add_trace(go.Scatter(
                    x=df_ipc_data['año_mes'],
                    y=df_ipc_data['var_ipc'],
                    name='IPC', 
                    mode='lines',
                    line=dict(color='white', dash='dot'), # Cambié a negro para asegurar visibilidad
                    showlegend=mostrar_leyenda,     
                ),
                row=1, col=col_num)
            
            fig14_ipc.update_layout(legend_title_text='Variación')
            fig14_ipc.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")
                
            tab1, tab2 = st.tabs(["Evolutivo CM ",'Variación vs IPC'])
            with tab1:
                st.plotly_chart(fig14, use_container_width=True)
            with tab2:
                st.plotly_chart(fig14_ipc, use_container_width=True)

            
# ----- GRAFICOS AJUSTADOS POR IPC --------------------------------------------------
        elif st.session_state['selected_variation_type'] == "IPC":

            # gráfico 1: evolución costo repuestos por tva IPC
            st.subheader('1. Evolución del costo prom. por TVA - Ajust. por IPC')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_rep_tv[['tva','año_mes','cant_ocompra','cant_piezas_total','var_cant_piezas',
                                        'cant_piezas_prom','monto_total_compras','ipc','monto_ipc','costo_prom_ipc','var_costo_prom_ipc']], hide_index=True,)

            fig7 = create_plot_orion(df_rep_tv, 'costo_prom_ipc', 'tva', None, 'Costo Promedio Ajust. por IPC')
            st.plotly_chart(fig7, use_container_width=True)
            st.markdown("---")
# ==========================================================================

            # gráfico 2: evolución costo repuestos por tipo repuesto IPC
            st.subheader('2. Evolución del costo prom. por Tipo Repuesto - Ajust. por IPC')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_tipo_rep[['año', 'año_mes', 'cant_ocompra', 'cant_piezas_total',
                            'cant_piezas_prom', 'ipc', 'monto_ipc', 'costo_prom_ipc',
                            'var_costo_prom_ipc', 'tipo_repuesto']], hide_index=True)

            fig8 = create_plot_orion(df_tipo_rep, 'costo_prom_ipc', 'tipo_repuesto', None,'Costo Promedio ajust. por IPC')
            st.plotly_chart(fig8, use_container_width=True)
            st.markdown("---")

            # muestro grafico torta MARCA AUTOS
            if st.button("Mostrar/Ocultar Distribución de Marcas Autos", icon='📊'):
                st.session_state.show_pie_chart = not st.session_state.show_pie_chart
            
            if st.session_state.show_pie_chart:
                st.subheader('Distribución de Órdenes de Compra por Marca')
                fig_pie = create_pie_chart(df_marcas_autos, 'cant_ocompra')
                st.plotly_chart(fig_pie, use_container_width=True)
                # st.text('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
                st.text('Total marcas: 44' )
                st.markdown("---")
# ==========================================================================

            # gráfico 3: evolución costo repuestos por marca autos IPC
            st.subheader('3. Costo de piezas prom. por Marca (autos) - Ajust. por IPC')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                st.dataframe(df_rtos_marca_mes[['marca','año_mes','cant_ocompra','cant_piezas_total',
                                        'costo_prom_ipc','var_costo_prom_ipc','monto_ipc']], hide_index=True,
                                        column_config=({
                                            "var_costo_prom_ipc": st.column_config.NumberColumn(
                                        "var_costo_prom_ipc", 
                                        format="%.2f",
                                    )}))

            fig18 = create_plot_orion(df_rtos_marca_mes, 'costo_prom_ipc', 'marca', None, 'Costo Promedio')
            st.plotly_chart(fig18, use_container_width=True)
            st.markdown("---")

            # muestro grafico torta MARCA CAMIONES
            if st.button("Mostrar/Ocultar Distribución de Marcas Camiones", icon='📊'):
                st.session_state.show_pie_chart_2 = not st.session_state.show_pie_chart_2
            
            if st.session_state.show_pie_chart_2:
                st.subheader('Distribución de Órdenes de Compra por Marca')
                fig_pie = create_pie_chart(df_marcas_camiones, 'cant_ocompra')
                st.plotly_chart(fig_pie, use_container_width=True)
                # st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
                st.text('Total marcas: 26')
                st.markdown("---")
# ==========================================================================

            # gráfico 4: evolución costo repuestos por marca camiones IPC
            st.subheader('4. Costo de piezas prom. por Marca (camiones) - Ajust. por IPC')

            # muestro el dataset 
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                st.dataframe(df_rtos_marca_mes_cam[['marca','año_mes','cant_ocompra','cant_piezas_total',
                                        'costo_prom_ipc','var_costo_prom_ipc','monto_ipc']], hide_index=True,)

            fig21 = create_plot_orion(df_rtos_marca_mes_cam, 'costo_prom_ipc', 'marca', None, 'Costo Promedio')
            st.plotly_chart(fig21, use_container_width=True)
            st.markdown("---")    
# ==========================================================================

# ----- GRAFICO 5: evolución costo mano de obra por tva y tipo de mano de obra IPC
            st.subheader('5. Evolución del costo de mano de obra prom. por Tipo de M.O y TVA - Ajust. por IPC')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                st.dataframe(df_cm_mo[['tva','año_mes','tipo_costo','valor_costo_hist','ipc','valor_costo_ipc',
                                    'var_ipc']], hide_index=True,)

            df_cm_mo = df_cm_mo[
                (df_cm_mo['tva'] == 'moto') & (df_cm_mo['tipo_costo'] != 'cm_hs_elect') |
                (df_cm_mo['tva'] != 'moto')
            ]
            fig12 = create_plot_orion(df_cm_mo, 'valor_costo_ipc', 'tva','tipo_costo', 'Costo Promedio ajust. por IPC')
            fig12.update_traces(
                    mode="lines+markers",
                    marker=dict(size=1),)
            st.plotly_chart(fig12, use_container_width=True)
            st.markdown("---")
# ==========================================================================

# ----- GRAFICO 6: evolución costo mano de obra cleas si vs cleas no IPC
            st.subheader('6. Comparativa variación M.O - CLEAS SI vs CLEAS NO - Ajust. por IPC')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                st.dataframe(df_cm_mo_cleas, hide_index=True,
                            column_config={
                                "var_costo": st.column_config.NumberColumn(
                                    "Var. Costo %", 
                                    format="%.2f",
                                ),
                                "var_ipc": st.column_config.NumberColumn(
                                    "Var. CM IPC %", 
                                    format="%.2f",
                                ),
                                "var_costo_usd": st.column_config.NumberColumn(
                                    "Var. CM en USD %", 
                                    format="%.2f",
                                    ),
                                }) 

            # quito camion_cleas_si del df resumen por poca cantidad de datos
            df_cm_mo_cleas = df_cm_mo_cleas[df_cm_mo_cleas['tva'] != 'camion_cleas_si']
            fig15 = create_plot_orion(df_cm_mo_cleas, 'valor_ipc', 'tva','tipo_costo', 'Costo Promedio ajust. por IPC')
            fig15.update_traces(
                    mode="lines+markers",
                    marker=dict(size=1),)
            st.plotly_chart(fig15, use_container_width=True)

        # ----- GRAFICOS EN USD -----
        elif st.session_state['selected_variation_type'] == "USD":

            # gráfico 1: evolución costo repuestos por tva USD    
            st.subheader('1. Evolución del costo prom. por TVA en USD')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_rep_tv[['tva','año_mes','cant_ocompra','cant_piezas_total','var_cant_piezas',
                                        'cant_piezas_prom','monto_total_compras','usd_blue','monto_usd','costo_prom_usd','var_costo_prom_usd']], hide_index=True,)

            fig9 = create_plot_orion(df_rep_tv, 'costo_prom_usd', 'tva', None, 'Costo Promedio (USD)')
            st.plotly_chart(fig9, use_container_width=True)
            st.markdown("---")
# ==========================================================================

            # gráfico 2: evolución costo repuestos por tipo repuesto USD
            st.subheader('2. Evolución del costo prom. por Tipo Repuesto en USD')

            # AGREGAR GRAF TORTA POR VALOR USD

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_tipo_rep[['año', 'año_mes', 'cant_ocompra', 'cant_piezas_total',
                            'cant_piezas_prom', 'usd_blue', 'monto_usd', 'costo_prom_usd',
                            'var_costo_prom_usd', 'tipo_repuesto']], hide_index=True)

            fig10 = create_plot_orion(df_tipo_rep, 'costo_prom_usd', 'tipo_repuesto', None, 'Costo Promedio (USD)')
            st.plotly_chart(fig10, use_container_width=True)
            st.markdown("---")

            # muestro grafico torta MARCA AUTOS
            if st.button("Mostrar/Ocultar Distribución de Marcas Autos" ,icon='📊'):
                st.session_state.show_pie_chart = not st.session_state.show_pie_chart
            
            if st.session_state.show_pie_chart:
                st.subheader('Distribución de Órdenes de Compra por Marca')
                fig_pie = create_pie_chart(df_marcas_autos, 'cant_ocompra')
                st.plotly_chart(fig_pie, use_container_width=True)
                # st.text('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
                st.text('Total marcas: 44' )
                st.markdown("---")
# ==========================================================================

            # gráfico 3: evolución costo repuestos por marca autos USD
            st.subheader('3. Costo de piezas prom. histórico por Marca (autos) en USD')

            # muestro el dataset
            with st.expander("Ver tabla de datos",icon=":material/query_stats:"):
                st.dataframe(df_rtos_marca_mes[['marca','año_mes','cant_ocompra','cant_piezas_total', 'usd_blue',
                                        'costo_prom_usd','var_costo_prom_usd','monto_usd']], hide_index=True,
                                        column_config={
                                        "var_costo_prom_usd": st.column_config.NumberColumn(
                                        "var_costo_prom_usd", 
                                        format="%.2f",
                                    )})

            fig19 = create_plot_orion(df_rtos_marca_mes, 'costo_prom_usd', 'marca', None, 'Costo Promedio (USD)')
            st.plotly_chart(fig19, use_container_width=True)
            st.markdown("---")

            # muestro grafico torta MARCA CAMIONES
            if st.button("Mostrar/Ocultar Distribución de Marcas Camiones", icon='📊'):
                st.session_state.show_pie_chart_2 = not st.session_state.show_pie_chart_2
            
            if st.session_state.show_pie_chart_2:
                st.subheader('Distribución de Órdenes de Compra por Marca')
                fig_pie = create_pie_chart(df_marcas_camiones, 'cant_ocompra')
                st.plotly_chart(fig_pie, use_container_width=True)
                # st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
                st.text('Total marcas: 26')
                st.markdown("---")     
# ==========================================================================

            # gráfico 4: evolución costo repuestos por marca camiones USD
            st.subheader('4. Costo de piezas prom. por Marca (camiones) en USD')

            # muestro el dataset
            with st.expander("Ver tabla de datos",icon=":material/query_stats:"):
                st.dataframe(df_rtos_marca_mes_cam[['marca','año_mes','cant_ocompra','cant_piezas_total', 'usd_blue',
                                        'costo_prom_usd','var_costo_prom_usd','monto_usd']], hide_index=True,)

            fig22 = create_plot_orion(df_rtos_marca_mes_cam, 'costo_prom_usd', 'marca', None, 'Costo Promedio (USD)')
            st.plotly_chart(fig22, use_container_width=True)
            st.markdown("---") 
# ==========================================================================

            # gráfico 5: evolución costo mano de obra por tva y tipo de mano de obra USD
            st.subheader('5. Evolución del costo de Mano de Obra prom. por Tipo de M.O y TVA en USD')

            # muestro el dataset
            with st.expander("Ver tabla de datos",icon=":material/query_stats:"):
                st.dataframe(df_cm_mo[['tva','año_mes','tipo_costo','valor_costo_hist','usd_blue','valor_costo_usd',
                                    'var_costo_usd']], hide_index=True,)

            df_cm_mo = df_cm_mo[
                (df_cm_mo['tva'] == 'moto') & (df_cm_mo['tipo_costo'] != 'cm_hs_elect') |
                (df_cm_mo['tva'] != 'moto')
            ]
            fig13 = create_plot_orion(df_cm_mo, 'valor_costo_usd', 'tva','tipo_costo', 'Costo Promedio (USD)')
            fig13.update_traces(
                    mode="lines+markers",
                    marker=dict(size=1),)
            st.plotly_chart(fig13, use_container_width=True)
            st.markdown("---")
# ==========================================================================

            # gráfico 6: evolución costo mano de obra cleas si vs cleas no USD
            st.subheader('6. Comparativa variación M.O en USD - CLEAS SI vs CLEAS NO')

            # muestro el dataset
            with st.expander("Ver tabla de datos",icon=":material/query_stats:"):
                st.dataframe(df_cm_mo_cleas, hide_index=True,
                                column_config={
                                "var_costo": st.column_config.NumberColumn(
                                    "Var. Costo %", 
                                    format="%.2f",
                                ),
                                "var_ipc": st.column_config.NumberColumn(
                                    "Var. CM IPC %", 
                                    format="%.2f",
                                ),
                                "var_costo_usd": st.column_config.NumberColumn(
                                    "Var. CM en USD %", 
                                    format="%.2f",
                                    ),
                                }) 

            # quito camion_cleas_si del df resumen por poca cantidad de datos
            df_cm_mo_cleas = df_cm_mo_cleas[df_cm_mo_cleas['tva'] != 'camion_cleas_si']
            fig16 = create_plot_orion(df_cm_mo_cleas, 'valor_usd', 'tva','tipo_costo', 'Costo Promedio (USD)')
            fig16.update_traces(
                    mode="lines+markers",
                    marker=dict(size=1),)
            st.plotly_chart(fig16, use_container_width=True)

        '''Se descarta línea del gráfico de 'camion_cleas_si' por poca cantidad de datos'''
# ==========================================================================
# ---- Análisis por PROVINCIA ----------------------------------------------
# ==========================================================================
    elif current_analysis == opcion_4:
        st.title('Análisis Coste Medio por Provincia')     
        st.markdown("---")   
        st.header('Coste Medio de repuestos por provincia')
        st.markdown("#### _Fuente de datos: Orion/Cesvi_ \n Actualización: **enero 2025**")
        

        def create_map_chart(df, color, selected_fecha):
            df_cm_filtered = df[df['año'] == selected_fecha]

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
                labels={'coste_medio': 'Coste Medio Promedio'},
                projection="mercator",
                width=1000, 
                height=1000  
            )

            hover_tmpl = "<b>%{location}</b><br>"
            hover_tmpl += "Coste medio Siniestral: $%{z:,.0f}<br><br>"
            fig.update_traces(hovertemplate=hover_tmpl)

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

        def create_map_chart_orion(df, selected_fecha):

            df_filtered = df[df['año'] == selected_fecha].copy()

            if df_filtered.empty:
                st.warning(f"No hay información para el año {selected_fecha}.")
                return None

            df_general = df_filtered[df_filtered['tipo_repuesto'] == 'general'][['provincia', 'costo_pieza_prom']]
            df_general.columns = ['provincia', 'costo_prom_total_real']

            df_detalles = df_filtered[df_filtered['tipo_repuesto'] != 'general']
            
            df_pivot = df_detalles.pivot_table(
                index='provincia', 
                columns='tipo_repuesto', 
                values='costo_pieza_prom',
                aggfunc='mean'
            ).reset_index()

            df_final = pd.merge(df_general, df_pivot, on='provincia', how='left')

            repuestos_cols = [c for c in df_pivot.columns if c != 'provincia']

            # --- Configuración de Escala ---
            min_cost = df_final['costo_prom_total_real'].min()
            max_cost = df_final['costo_prom_total_real'].max()
            ROUNDING_UNIT = 50000
            if not np.isnan(min_cost):
                min_cost = np.floor(min_cost / ROUNDING_UNIT) * ROUNDING_UNIT
            else:
                min_cost, max_cost = 0, 1

            # MAPA
            fig = px.choropleth(
                df_final,
                geojson=provincias_geojson,
                locations='provincia',
                featureidkey="properties.nombre_normalizado",
                color='costo_prom_total_real', # Ahora el color lo da la categoría 'general'
                color_continuous_scale="oranges",
                range_color=[min_cost, max_cost],
                # Cargamos los repuestos específicos en custom_data para el hover
                hover_data={col: ":,.0f" for col in repuestos_cols},
                labels={'costo_prom_total_real': 'Costo Medio Gral.'},
                projection="mercator",
                width=1000,
                height=800
            )

            hover_tmpl = "<b>%{location}</b><br>"
            hover_tmpl += "Coste Medio general: $%{z:,.0f}<br><br>" # %{z} toma el valor de 'color'
            
            # Añadimos cada repuesto específico al cartelito
            for col in repuestos_cols:
                nombre_limpio = col.replace('_', ' ').title()
                # Buscamos el índice correspondiente en hover_data/customdata
                hover_tmpl += f"{nombre_limpio}: $ %{{customdata[{repuestos_cols.index(col)}]}}<br>"
            
            fig.update_traces(hovertemplate=hover_tmpl)

            fig.update_geos(
                visible=False,
                fitbounds=False,
                showcountries=True,
                landcolor="lightgrey",
                showland=True,
                scope="south america",
                projection_scale=1,
            )

            return fig

# ----- Comparativo Orion/Cesvi por provincia --------------------------------------------------
        available_fechas = sorted(df_cm_prov_orion['año'].unique().tolist())

        # 2 cols para separar grafico y contenedor de filtros
        col3, col4 = st.columns([1, 4], gap='large') # la segunda col es 4 veces el ancho de la primera 
        
        with col3:  
            with st.container(border=True):
                # contenedor para seleccionar fecha
                selected_fecha = st.selectbox(
                    "Seleccionar año:",
                    options=available_fechas,  
                    index=len(available_fechas)-1 
                )

        with col4:
            with st.container(border=True):            
                st.markdown(f"#### Año: {selected_fecha} \n #### Coverable: AUT")
                fig_prov = create_map_chart_orion(df_cm_prov_orion, selected_fecha)
                st.plotly_chart(fig_prov, use_container_width=False)    
        
        st.markdown("#### Tabla comparativa: Coste Medio por provincia y repuesto - Orion/Cesvi")  
        st.dataframe(comparativo_orion_prov, 
                     use_container_width=True,
                     hide_index=True,
                     column_config={
                        'provincia': 'Provincia',
                        'tipo_repuesto': 'Tipo de Repuesto',
                        "cant_ord_compra_2024": st.column_config.NumberColumn("Cant ord. 2024", format="%d"),
                        "cant_ord_compra_2025": st.column_config.NumberColumn("Cant ord. 2025", format="%d"),
                        "costo_pieza_prom_2024": st.column_config.NumberColumn("Costo prom. 2024", format="$ %.0f"),
                        "costo_pieza_prom_2025": st.column_config.NumberColumn("Costo prom. 2025", format="$ %.0f"),
                        "var_cant_ord": st.column_config.NumberColumn(
                            "Var. Cant %", 
                            format="%.1f%%",
                            help="Variación en la cantidad de órdenes"
                        ),
                        "var_costo_prom": st.column_config.NumberColumn(
                            "Var. Costo %", 
                            format="%.1f%%",
                            help="Variación en el costo promedio de la pieza"
                            )
                        })    

# ==========================================================================

# ----- Comparativo BI La Segunda por provincia --------------------------------------------------
        st.header('Coste Medio siniestral por provincia')
        st.markdown("#### _Fuente de datos: BI La Segunda_")

        available_fechas = sorted(df_cm_prov['año'].unique().tolist())

        # 2 cols para separar grafico y contenedor de filtros
        col1, col2 = st.columns([1, 4], gap='large') # la segunda col es 4 veces el ancho de la primera 
        
        with col1:  
            with st.container(border=True):
        # contenedor para seleccionar fecha
                selected_fecha = st.selectbox(
                "Seleccionar año:",
                options=available_fechas,   
                index=len(available_fechas)-1, 
                key="fecha_analisis_provincias"
                )


        with col2:
            with st.container(border=True):
                # st.subheader(f'Análisis Coste Medio por Provincia - {selected_coverable_map}')
                st.markdown(f"#### Coverable: AUT")
                st.markdown(f"#### Año: {selected_fecha}")
                fig_prov = create_map_chart(df_cm_prov, 'coste_medio', selected_fecha)
                st.plotly_chart(fig_prov, use_container_width=False)    

        st.markdown("#### Tabla comparativa: Coste Medio siniestral por provincia")  
        st.dataframe(comparativo_cm_siniestral, 
                     use_container_width=True,
                     hide_index=True,
                    #  width=700,
                     column_config={
                         'provincia': 'Provincia',
                         "coste_medio_2024": st.column_config.NumberColumn("Coste medio 2024", format="$ %.0f"),
                         "coste_medio_2025": st.column_config.NumberColumn("Coste medio 2025", format="$ %.0f"),
                         "var_coste_medio": st.column_config.NumberColumn(
                            "Var. CM %", 
                            format="%.1f%%",
                            help="Variación coste medio siniestral (2024-2025)"
                         )})

        # with st.expander("Ver data cruda",icon=":material/query_stats:"):
        #     st.markdown("#### Data Cruda")
        #     # Para mostrar los datos crudos filtrados (opcional, ajusta tu lógica de datos)
        #     df_cm_filtered_raw = df_cm_prov[(df_cm_prov['coverable'] == selected_coverable_map) &
        #                                                     (df_cm_prov['año'] == selected_fecha)]
        #     st.dataframe(df_cm_filtered_raw, use_container_width=True)   

# ==========================================================================
# ----- Comparativo Mano de obra -------------------------------------------
# ==========================================================================

    elif current_analysis == opcion_5:
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
        
        def create_plot_mo(df, y_col, color, facet_col, y_label, leg_title_text='Aseguradora', x_tickangle=None, line_width=2):       
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
                color_discrete_sequence=["gray", "cyan", "#FB0D0D", '#2CA02C', "blue", "magenta", "orange", ],
                facet_col=facet_col,
                labels={'value': y_label, 'anio_mes': ''}
            )

            fig.update_traces(
                    mode="lines+markers",
                    marker=dict(size=2),
                    hovertemplate="<b>%{fullData.name}:</b> $ %{y:,.0f}<extra></extra>" # Muestra solo el valor con separador de miles
                )
            
            fig.update_layout(
                legend_title_text=leg_title_text,
                height=400, # Altura del subplot individual
                font=dict(family="Arial", size=15),
                margin=dict(t=50, b=0, l=0, r=0),
                hovermode="x unified",
            )
            fig.for_each_xaxis(
                lambda xaxis: xaxis.update(
                    tickangle=x_tickangle, # Aplicar el ángulo deseado
                    showticklabels=True     # Asegurarse de que las etiquetas sean visibles (si fuera necesario)
                )
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
                color_discrete_sequence=["gray", "cyan", "#FB0D0D", '#2CA02C', "blue", "magenta", "orange"],
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
            y_cols_hist = ['grupo_cesvi', 'grupo_sls', 'la_segunda']
            
            st.subheader('Evolución monto de Repuestos y Mano de Obra (MO)')
            # mostrar evolutivo MO (Chapa/Pintura)
            if st.button("Mostrar/Ocultar Evolutivo M.O (chapa y pintura)", icon='📈'):
                st.session_state.show_mo = not st.session_state.show_mo
            
            if st.session_state.show_mo:
                st.markdown('#### Evolutivo chapa y pintura')
                fig_mo = create_plot_mo_area(df_chapa_pintura, 'monto_historico', 'aseguradora', 'tipo', 'Monto')
                

                with st.expander("Ver tabla de datos",icon=":material/query_stats:"):
                    # st.subheader("Tabla de Datos de Ejemplo")
                    st.dataframe(df_chapa_pintura[['anio_mes','aseguradora','monto_historico','tipo']], hide_index=True, width=1500,)

                st.plotly_chart(fig_mo, use_container_width=True)
                st.subheader('', divider='grey') 
                st.markdown("")
            
            fig_1 = create_plot_mo(df_mo_repuestos_final, 'monto_historico', 'aseguradora', 'tipo', 'Monto', x_tickangle=45)

            # muestro el dataset
            with st.expander("Ver tabla de datos",icon=":material/query_stats:"):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_mo_repuestos_final[['anio_mes','aseguradora','tipo','monto_historico','var_monto_prom_%','ipc','ipc_empalme_ipim','var_ipc_%',]],
                            hide_index=True, width=1000,)

            df_ipc_data_mo = df_mo_repuestos_final[['anio_mes', 'var_ipc']].drop_duplicates().sort_values('anio_mes')

            fig_1_ipc = create_plot_mo(df_mo_repuestos_final, 'var_monto_prom', 'aseguradora', 'tipo','Variación (base 1)', x_tickangle=45)
            fig_1_ipc.add_trace(go.Scatter(
                x=df_ipc_data_mo['anio_mes'],
                y=df_ipc_data_mo['var_ipc'],
                name='IPC', 
                mode='lines',
                line=dict(color='white', dash='dot'), # Cambié a negro para asegurar visibilidad
                showlegend=True,     
            ),
            row=1, col=1)

            fig_1_ipc.add_trace(go.Scatter(
                x=df_ipc_data_mo['anio_mes'],
                y=df_ipc_data_mo['var_ipc'],
                name='IPC',        # Nombre que aparecerá en la leyenda
                mode='lines',
                line=dict(color='white', dash='dot'),
                showlegend=False, # Importante: Ocultar esta traza de la leyenda 
            ),
            row=1, col=2)
            fig_1_ipc.update_layout(legend_title_text='Variación')
            fig_1_ipc.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")


            tab1, tab2 = st.tabs(["Evolutivo CM ",'Variación vs IPC'])
            with tab1:
                st.plotly_chart(fig_1, use_container_width=True)
            with tab2:
                st.plotly_chart(fig_1_ipc, use_container_width=True)


            st.subheader('', divider='grey') 
# ==========================================================================

            st.subheader('Evolución monto de reparaciones (Repuestos + MO)')
            
            fig_3 = create_plot_mo(df_tot_reparacion, y_cols_hist, None, None, 'Monto MO')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_tot_reparacion[['anio_mes', 'grupo_cesvi', 'grupo_sls', 'la_segunda',
                                         'ipc','ipc_empalme_ipim','var_ipc', 'var_%_grupo_cesvi', 'var_%_grupo_sls', 'var_%_la_segunda',
                                        ]], hide_index=True,)

            y_cols_var = ['var_%_grupo_cesvi', 'var_%_grupo_sls', 'var_%_la_segunda']

            fig_3_ipc = create_plot_mo(df_tot_reparacion, y_cols_var, None, None, 'Variación (base 1)')
            fig_3_ipc.add_trace(go.Scatter(
                x=df_tot_reparacion['anio_mes'],
                y=df_tot_reparacion['var_ipc'],
                name='var_ipc', 
                mode='lines',
                line=dict(color='white', dash='dot'),
            ))
            fig_3_ipc.update_layout(legend_title_text='')
            fig_3_ipc.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")

            tab1, tab2 = st.tabs(["Evolutivo CM ",'Variación vs IPC'])
            with tab1:
                st.plotly_chart(fig_3, use_container_width=True)
            with tab2:
                st.plotly_chart(fig_3_ipc, use_container_width=True)

            st.subheader('', divider='grey') 

# ==========================================================================
            st.subheader('Evolución costo hora de Mano de Obra')
            fig_5 = create_plot_mo(df_costo_hora, y_cols_hist, None, None, 'Costo hora')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_costo_hora[['anio_mes', 'grupo_cesvi', 'grupo_sls', 'la_segunda',
                                       'ipc','ipc_empalme_ipim','var_ipc', 'var_%_grupo_cesvi', 'var_%_grupo_sls', 'var_%_la_segunda',
                                        ]], hide_index=True,)

            fig_5_ipc = create_plot_mo(df_costo_hora, y_cols_var, None, None, 'Variación (base 1)')
            fig_5_ipc.add_trace(go.Scatter(
                x=df_costo_hora['anio_mes'],
                y=df_costo_hora['var_ipc'],
                name='var_ipc', 
                mode='lines',
                line=dict(color='white', dash='dot'),
            ))
            fig_5_ipc.update_layout(legend_title_text='')
            fig_5_ipc.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")

            tab1, tab2 = st.tabs(["Evolutivo CM ",'Variación vs IPC'])
            with tab1:
                st.plotly_chart(fig_5, use_container_width=True)
            with tab2:
                st.plotly_chart(fig_5_ipc, use_container_width=True)

# ==========================================================================
            st.markdown('')
            st.subheader('Peritaciones', divider='grey')
            
            st.subheader('▫️ Evolución cantidad de Peritaciones')
            fig_4 = create_plot_mo(df_peritaciones, y_cols_hist, None, None, 'Cantidad de Peritaciones', leg_title_text='')
            st.plotly_chart(fig_4, use_container_width=True)

            # muestro el dataset
            with st.expander("Ver tabla de datos",):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_peritaciones[['anio_mes', 'grupo_cesvi', 'grupo_sls', 'la_segunda',]], hide_index=True, width=1000,)
# ==========================================================================

            st.subheader('▫️ % Variación mensual de cantidad de Peritaciones')
            y_var=['var_%_grupo_cesvi', 'var_%_grupo_sls', 'var_%_la_segunda']
            fig_6 = create_plot_mo(df_peritaciones, y_var, None, None, '% variación', leg_title_text='')
            fig_6.update_traces(hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>")
            
            st.plotly_chart(fig_6, use_container_width=True)

            # muestro el dataset
            with st.expander("Ver tabla de datos",):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_peritaciones[['anio_mes', 'part_grupo_sls_vs_cesvi', 'part_la_segunda_vs_cesvi']], 
                            hide_index=True, width=1000,)
# ==========================================================================
             
            # st.subheader('▫️ % Participacion respecto a Grupo Cesvi')
            # y_cols_part=['part_grupo_sls_vs_cesvi', 'part_la_segunda_vs_cesvi' ]
            # fig_6 = create_plot_mo(df_peritaciones, y_cols_part, None, None, '% participacion', leg_title_text='')
            # st.plotly_chart(fig_6, use_container_width=True)

            # # muestro el dataset
            # with st.expander("Ver tabla de datos",):
            #     # st.subheader("Tabla de Datos de Ejemplo")
            #     st.dataframe(df_peritaciones[['anio_mes', 'var_%_grupo_cesvi', 'var_%_grupo_sls', 'var_%_la_segunda']], 
            #                 hide_index=True, width=1000,)

# ----- GRAFICOS IPC --------------------------------------------------
        if st.session_state['selected_variation_type_2'] == "IPC":
            y_cols_ipc = ['grupo_cesvi_ipc', 'grupo_sls_ipc', 'la_segunda_ipc']
            
            st.subheader('Evolución monto de Repuestos y Mano de Obra (MO) - ajust. por IPC')
            # mostrar evolutivo MO (Chapa/Pintura)
            if st.button("Mostrar/Ocultar Evolutivo M.O (chapa y pintura)", icon='📈'):
                st.session_state.show_mo = not st.session_state.show_mo
            
            if st.session_state.show_mo:
                st.markdown('#### Evolutivo chapa y pintura IPC')
                with st.expander("Ver tabla de datos", icon=':material/query_stats:'):
                    # st.subheader("Tabla de Datos de Ejemplo")
                    st.dataframe(df_chapa_pintura[['anio_mes','aseguradora','monto_ipc','tipo']], hide_index=True, width=1500,)
                fig_mo = create_plot_mo_area(df_chapa_pintura, 'monto_ipc', 'aseguradora', 'tipo', 'Monto')
                st.plotly_chart(fig_mo, use_container_width=True)

            fig_1 = create_plot_mo(df_mo_repuestos_final, 'monto_ipc', 'aseguradora', 'tipo', 'Monto')
            st.plotly_chart(fig_1, use_container_width=True)

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=':material/query_stats:'):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_mo_repuestos_final[['anio_mes','aseguradora','monto_ipc','tipo']], hide_index=True, width=1500,)
            
            st.subheader('', divider='grey') 

            st.subheader('Evolución monto de reparaciones (Repuestos + MO) - ajust. por IPC')

            fig_3 = create_plot_mo(df_tot_reparacion, y_cols_ipc, None, None, 'Monto MO')
            st.plotly_chart(fig_3, use_container_width=True)
            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=':material/query_stats:'):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_tot_reparacion[['anio_mes','ipc','grupo_cesvi_ipc', 'grupo_sls_ipc', 'la_segunda_ipc']], 
                            hide_index=True, width=1000,)

            st.subheader('', divider='grey') 
        
            st.subheader('Evolución Costo Hora de Mano de Obra - ajust. por IPC')
            fig_5 = create_plot_mo(df_costo_hora, y_cols_ipc, None, None, 'Costo hora')
            st.plotly_chart(fig_5, use_container_width=True)

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=':material/query_stats:'):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_costo_hora[['anio_mes','ipc','grupo_cesvi_ipc', 'grupo_sls_ipc', 'la_segunda_ipc']], hide_index=True,)

# ----- GRAFICOS USD --------------------------------------------------
        if st.session_state['selected_variation_type_2'] == "USD":
            y_cols_usd = ['grupo_cesvi_usd', 'grupo_sls_usd', 'la_segunda_usd']
            
            st.subheader('Evolución monto de Repuestos y Mano de Obra (MO) - en USD')
            # mostrar evolutivo MO (Chapa/Pintura)
            if st.button("Mostrar/Ocultar Evolutivo M.O (chapa y pintura)", icon='📈'):
                st.session_state.show_mo = not st.session_state.show_mo
            
            if st.session_state.show_mo:
                st.markdown('#### Evolutivo chapa y pintura en USD')
                with st.expander("Ver tabla de datos", icon=':material/query_stats:'):
                    # st.subheader("Tabla de Datos de Ejemplo")
                    st.dataframe(df_chapa_pintura[['anio_mes','aseguradora','monto_usd','tipo']], hide_index=True, width=1500,)
                fig_mo = create_plot_mo_area(df_chapa_pintura, 'monto_usd', 'aseguradora', 'tipo', 'Monto')
                st.plotly_chart(fig_mo, use_container_width=True)


            fig_1 = create_plot_mo(df_mo_repuestos_final, 'monto_usd', 'aseguradora', 'tipo', 'Monto')
            st.plotly_chart(fig_1, use_container_width=True)

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=':material/query_stats:'):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_mo_repuestos_final[['anio_mes','aseguradora','monto_usd','tipo']], hide_index=True, width=1500,)

            st.subheader('', divider='grey') 

            st.subheader('Evolución monto de reparaciones (Repuestos + MO) - en USD')

            fig_3 = create_plot_mo(df_tot_reparacion, y_cols_usd, None, None, 'Monto MO')
            st.plotly_chart(fig_3, use_container_width=True)
            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=':material/query_stats:'):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_tot_reparacion[['anio_mes','usd_blue','grupo_cesvi_usd', 'grupo_sls_usd', 'la_segunda_usd']], 
                            hide_index=True, width=1000,)
            
            st.subheader('', divider='grey') 

            st.subheader('Evolución Costo Hora de Mano de Obra - en USD')
            fig_5 = create_plot_mo(df_costo_hora, y_cols_usd, None, None, 'Costo hora')
            st.plotly_chart(fig_5, use_container_width=True)
            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=':material/query_stats:'):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_costo_hora[['anio_mes','usd_blue','grupo_cesvi_usd', 'grupo_sls_usd', 'la_segunda_usd']], 
                            hide_index=True, width=1000,)

# ==========================================================================
# EVOLUTIVO PAGOS CRISTALES L2
# ==========================================================================

    elif current_analysis == opcion_pagos_pkt:
        st.markdown('## Evolución de monto de pagos de Cristales (L2)')    
        st.markdown("---")
        st.markdown("#### _Fuente de datos:_ \
                    \n:white_small_square: _La Segunda BI (pagos)_")
        
        st.markdown('---')

        def create_plot_pagos(df_source, y1, y2, y3, title, x_tickangle=45):

            df_filtered = df_source[
                (df_source['marca_vehiculo'].isin(final_marcas_to_filter)) 
                & (df_source['tipo_cristal'] == selected_tipo_cristal)
            ].copy()

            df_filtered.sort_values('año_mes_fecha_pago', inplace=True)
            df_plot = df_filtered.groupby('año_mes_fecha_pago').agg(
                {
                y1: 'mean',
                y2: 'mean',           
                y3: 'mean'         
                }).reset_index()

            # Columnas y etiquetas específicas para el gráfico (ajustadas al gráfico de la imagen)
            y1_cols = [y1, y2] # Eje ARS (Primario)
            y2_cols = [y3]                        # Eje USD (Secundario)
            
            y1_label = "Monto (ARS)"
            y2_label = "Monto (USD)"
            x_col = 'año_mes_fecha_pago'


            line_colors = {
                y1: '#1f77b4', 
                y2: '#ff7f0e',       
                y3: '#2ca02c'           
            }
            legend_names = {
                y1: y1,
                y2: y2,
                y3: y3
            }

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            #  eje primario
            for col in y1_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot[x_col], 
                        y=df_plot[col], 
                        name=legend_names[col],
                        line=dict(color=line_colors[col], width=3), 
                        showlegend=True,
                        # hovertemplate=f"{legend_names[col]}: %{{y:.2f}} ARS<extra></extra>"
                    ),
                    secondary_y=False, # Eje Y Izquierdo
                )

            # eje secundario
            for col in y2_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot[x_col], 
                        y=df_plot[col], 
                        name=legend_names[col],
                        line=dict(color=line_colors[col], width=3),
                        showlegend=True,
                        # hovertemplate=f"{legend_names[col]}: %{{y:.2f}} USD<extra></extra>"
                    ),
                    secondary_y=True, # Eje Y Derecho
                )


            fig.update_yaxes(title_text=f"{y1_label}", secondary_y=False, nticks=14, showgrid=True)
            fig.update_yaxes(title_text=f"{y2_label}", secondary_y=True, nticks=14, showgrid=False)

            # Ajustes del Gráfico
            fig.update_layout(
                title_text=title,
                height=700,
                legend_title_text='', # Dejar vacío ya que el nombre de la línea lo explica
                font=dict(family="Arial", size=15),
                margin=dict(t=100, b=0, l=0, r=0),
                title=dict(
                    font=dict(size=20, family="Arial"),
                ),
            )

            # eje X
            fig.update_xaxes(
                tickangle=x_tickangle, 
                showticklabels=True,
                title_text=''
            )
                    
            return fig
        
        def create_plot_pagos_marcas(df, y_col, color, facet_col, y_label, title, x_tickangle=None):

            if df.empty:
                fig = go.Figure().update_layout(
                    title_text=f"No hay datos para graficar",
                    height=400,
                    font=dict(family="Arial", size=10),
                    title_font_size=12
                )
                return fig

            df_filtered = df[
                (df['tipo_cristal'] == selected_tipo_cristal) &
                (df['marca_vehiculo'].isin(final_marcas_to_filter)) 
                # & (df['zona'].isin(selected_zone))
            ].copy()

            df_filtered.sort_values('año_mes_fecha_pago', inplace=True)
            df_plot = df_filtered.groupby(['año_mes_fecha_pago', 'marca_vehiculo']).agg(
                {
                'monto_transaccion': 'mean',
                'pago_usd': 'mean',           
                'pago_ipc': 'mean'         
                }).reset_index()
            
            fig = px.line(
                df_plot,
                x='año_mes_fecha_pago',
                y=y_col,
                color=color,
                color_discrete_sequence=["#FB0D0D", "lightgreen", "blue", "gray", "magenta", "cyan", "orange", '#2CA02C'],
                facet_col=facet_col,
                labels={'año_mes_fecha_pago': '', y_col: y_label,}# 'marca': 'Marca', 'tipo_cristal': 'Tipo de Cristal'}
            )

            fig.update_layout(
                title_text=title,
                height=700, # Altura del subplot individual
                # width=200,
                legend_title_text='',
                font=dict(family="Arial", size=15),
                margin=dict(t=50, b=0, l=0, r=0),
                title=dict(
                    font=dict(size=20, family="Arial")       
            ),
                legend=dict(
                orientation="h",        # Muestra la leyenda horizontalmente
                yanchor="top",          # Anclamos la leyenda en la parte superior del espacio que le damos (y)
                y=-0.2,                 # Colocamos la leyenda debajo del gráfico (ajusta este valor si es necesario)
                xanchor="center",       # Anclamos la leyenda en su centro
                x=0.5)                   # Posicionamos el centro de la leyenda en el medio del eje X (0.5)
                )

            fig.for_each_xaxis(
            lambda xaxis: xaxis.update(
                tickangle=x_tickangle, # Aplicar el ángulo deseado
                showticklabels=True     # Asegurarse de que las etiquetas sean visibles (si fuera necesario)
                )
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig.update_traces(line=dict(width=2))
            
            return fig

        available_marcas = sorted(df_pagos_cristal['marca_vehiculo'].unique().tolist())
        available_marcas_todas = ["TODAS (general)"] + available_marcas
        available_tipo_cristal = sorted(df_pagos_cristal['tipo_cristal'].unique().tolist())
        DEFAULT_MARCAS = ["VOLKSWAGEN", "CHEVROLET", "FORD",  "TOYOTA", "FIAT", "PEUGEOT", "RENAULT"]



        # selectbox para tipo de vehiculo
        with st.sidebar:
            st.markdown("---")
            st.markdown("Filtros")
            st.markdown("##### _Seleccionar Tipo de Vehículo:_") 
            selected_tipo_cristal = st.selectbox(
                "Tipo cristal",
                options=available_tipo_cristal,
                index=0,
                label_visibility ='collapsed',
            )
            st.markdown("---")

            if selected_tipo_cristal:
                df_filtered_by_tipo_crist = df_pagos_cristal[df_pagos_cristal['tipo_cristal'] == selected_tipo_cristal]
                available_marcas_for_tipo_crist = sorted(df_filtered_by_tipo_crist['marca_vehiculo'].unique().tolist())
            else:
                available_marcas_for_tipo_crist = sorted(df_pagos_cristal['marca_vehiculo'].unique().tolist())

            default_selection = [
                m for m in DEFAULT_MARCAS 
                if m in available_marcas_for_tipo_crist
            ]

            # multiselect para diferentes marcas
            st.markdown("##### _Seleccionar Marcas:_")
            selected_marcas = st.multiselect(
                "Marcas",
                options=available_marcas_for_tipo_crist,
                default=default_selection,
                label_visibility='collapsed',
                placeholder="Seleccione una o varias Marcas..."
            )
            st.markdown("---")

            if not selected_marcas:
                final_marcas_to_filter = available_marcas_for_tipo_crist
                st.info(f"Filtro de Marcas: **TODAS** ({len(available_marcas_for_tipo_crist)} marcas)")
            else:
                final_marcas_to_filter = selected_marcas

# ----- PAGOS CRISTALES EVOLUTIVO --------------------------------------------------
        fig_pagos_cristal = create_plot_pagos(
            df_pagos_cristal, 
            'monto_transaccion',
            'pago_ipc',
            'pago_usd',
            title=f'Pagos promedio de {selected_tipo_cristal} (L2) ', 
            x_tickangle=45
        )
        st.plotly_chart(fig_pagos_cristal, use_container_width=True)
        
        df_pagos_cristal.sort_values('año_mes_fecha_pago', inplace=True)
        pagos_cristal_filtered = df_pagos_cristal[
            (df_pagos_cristal['marca_vehiculo'].isin(final_marcas_to_filter)) 
            & (df_pagos_cristal['tipo_cristal'] ==selected_tipo_cristal)
        ]
        pagos_cristal_filtered = pagos_cristal_filtered.groupby(['año_mes_fecha_pago']).agg(
            {'monto_transaccion': 'mean',
            'pago_ipc': 'mean',           
            'pago_usd': 'mean'}).reset_index()
        
        pagos_cristal_filtered.monto_transaccion = pagos_cristal_filtered.monto_transaccion.round(0).astype(int) 
        pagos_cristal_filtered.pago_usd = pagos_cristal_filtered.pago_usd.round(0).astype(int) 
        pagos_cristal_filtered.pago_ipc = pagos_cristal_filtered.pago_ipc.round(0).astype(int)

        with st.expander("Ver tabla de datos (resumen)", icon=":material/query_stats:"):
            st.dataframe(pagos_cristal_filtered, hide_index=True, width=900)

        st.subheader('', divider='grey')


# ----- PAGO CRISTALES POR MARCA --------------------------------------------------

        fig_pagos_cristal_hist = create_plot_pagos_marcas(
            df_pagos_cristal,
            'monto_transaccion', 
            'marca_vehiculo',
            None,
            'Monto histórico',
            title=f'Pagos robo de ruedas históricos por marca - {selected_tipo_cristal}', 
            x_tickangle=45)
        
        fig_pagos_cristal_ipc = create_plot_pagos_marcas(
            df_pagos_cristal,
            'pago_ipc', 
            'marca_vehiculo',
            None,
            'Monto IPC',
            title=f'Pagos robo de ruedas por marca ajustados por IPC - {selected_tipo_cristal}', 
            x_tickangle=45)
        
        fig_pagos_cristal_usd = create_plot_pagos_marcas(
            df_pagos_cristal,
            'pago_usd', 
            'marca_vehiculo',
            None,
            'Monto USD',
            title=f'Pagos robo de ruedas por marca en valor USD - {selected_tipo_cristal}', 
            x_tickangle=45)


        tab1, tab2, tab3 = st.tabs(["Histórico", "IPC", 'USD'])
        with tab1:
            st.plotly_chart(fig_pagos_cristal_hist, use_container_width=True)
        with tab2:
            st.plotly_chart(fig_pagos_cristal_ipc, use_container_width=True)
        with tab3:
            st.plotly_chart(fig_pagos_cristal_usd, use_container_width=True)
                    
        df_resumen_cristales = df_pagos_cristal.sort_values(by=['año_mes_fecha_pago', 'marca_vehiculo'])
        df_resumen_cristales = df_resumen_cristales[(df_resumen_cristales['tipo_cristal'] == selected_tipo_cristal)
                & (df_resumen_cristales['marca_vehiculo'].isin(final_marcas_to_filter))
            ]
        df_tabla_cristales = df_resumen_cristales.groupby(['año_mes_fecha_pago','marca_vehiculo']).agg(
            {'monto_transaccion': 'mean',
            'pago_usd': 'mean',           
            'pago_ipc': 'mean'}).reset_index()
        
        df_tabla_cristales.monto_transaccion = df_tabla_cristales.monto_transaccion.round(0).astype(int) 
        df_tabla_cristales.pago_usd = df_tabla_cristales.pago_usd.round(0).astype(int) 
        df_tabla_cristales.pago_ipc = df_tabla_cristales.pago_ipc.round(0).astype(int)

        with st.expander("Ver tabla de datos (resumen)", icon=":material/query_stats:"):
            st.dataframe(df_tabla_cristales, hide_index=True, width=900)

        st.subheader('', divider='grey')

        
# ----- COMPARATIVO PARTICIPACION DE MERCADO CRISTALES POR MARCA --------------------------

        column_1, column_2 = st.columns(2)

        with column_1:
            st.markdown(f'#### :white_small_square: Participación de pagos por Marca - {selected_tipo_cristal}')
            df_participacion = df_pagos_cristal.sort_values(by=['marca_vehiculo'])
            df_participacion = df_participacion[
                (df_participacion['tipo_cristal'] == selected_tipo_cristal)
                ]
            df_participacion = df_participacion.groupby(['marca_vehiculo']).agg(
                {'monto_transaccion': 'sum',}).reset_index()
            
            monto_total = df_participacion['monto_transaccion'].sum()
            df_participacion['% Participación'] = (
                df_participacion['monto_transaccion'] / monto_total
            ) * 100
            
            df_participacion['% Participación'] = df_participacion['% Participación'].round(0).astype(int)
            
            df_participacion.sort_values(
                '% Participación', 
                ascending=False, 
                inplace=True
            )
            df_participacion['Part. Acum.'] = df_participacion['% Participación'].cumsum()
            df_participacion['Part. Acum.'] = df_participacion['Part. Acum.'].round(0).astype(int)


            def format_participacion(row): 

                participacion_individual_str = f"{row['% Participación']} %"
                
                # corte en 90% de acumulacion de pagos
                if row['Part. Acum.'] > 90.00:
                    # Si supera el 90%, el acumulado es vacío, pero el individual se mantiene
                    participacion_acumulada_str = ''
                else:
                    # Si es <= 90%, formateamos el acumulado
                    participacion_acumulada_str = f"{row['Part. Acum.']} %" 

                return pd.Series(
                    [participacion_individual_str, participacion_acumulada_str], 
                    index=['% Participación', 'Part. Acum. (%)']
                )

            df_participacion[['% Participación', 'Part. Acum. (%)']] = df_participacion.apply(
                format_participacion, 
                axis=1 
            )
            df_participacion.drop(columns=['Part. Acum.'], inplace=True)


            fila_total = pd.DataFrame([{
                'marca_vehiculo': 'TOTAL',
                'monto_transaccion': monto_total,
                '% Participación': '',
                'Part. Acum. (%)': ''
            }])

            # Concatenar la fila total al final del DataFrame
            df_participacion = pd.concat(
                [df_participacion, fila_total],
                ignore_index=True
            )

            st.dataframe(df_participacion.rename(columns={'monto_transaccion':'Suma de Pagos'}), hide_index=True, width=450, height=600)

# ----- PARTICIPACION POR MODELO --------------------------------------------------
        with column_2:
            st.markdown('#### :white_small_square: Participación de pagos por Modelo')

            TARGET_BRAND = "TOYOTA"

            if TARGET_BRAND in available_marcas_for_tipo_crist:
                # Si TOYOTA existe, encontramos su índice en la lista FINAL (que tiene el placeholder en la pos 0)
                default_index = available_marcas_for_tipo_crist.index(TARGET_BRAND)
            else:

                default_index = 0

            selected_brand_for_model = st.selectbox(
                "Seleccionar Marca:",
                options=available_marcas_for_tipo_crist,
                index=default_index,
                label_visibility='visible',
                placeholder="Seleccione una Marca...",
            )

            if selected_brand_for_model and selected_tipo_cristal:
                # filtro por modelo
                df_modelos = df_pagos_cristal[
                    (df_pagos_cristal['marca_vehiculo'] == selected_brand_for_model) &
                    (df_pagos_cristal['tipo_cristal'] == selected_tipo_cristal)
                ].copy()
                
                df_modelos_participacion = df_modelos.groupby('modelo_vehiculo').agg(
                    {'monto_transaccion': 'sum'}
                ).reset_index()
                
                df_modelos_participacion.rename(
                    columns={'monto_transaccion': 'Suma de Pagos'}, 
                    inplace=True
                )

                df_modelos_participacion.sort_values(
                    'Suma de Pagos', 
                    ascending=False, 
                    inplace=True
                )
                monto_total_marca = df_modelos_participacion['Suma de Pagos'].sum()

                df_modelos_participacion['% Participación'] = (
                    df_modelos_participacion['Suma de Pagos'] / monto_total_marca
                ) * 100
                
                # Redondear y formatear el porcentaje
                df_modelos_participacion['% Participación'] = (
                    df_modelos_participacion['% Participación']
                    .round(0)
                    .astype(int)
                    .astype(str) + ' %'
                )

                st.dataframe(df_modelos_participacion, hide_index=True, width=450, height=600)

            else:
                st.info("Por favor, seleccione una marca de vehículo para ver la participación por modelo.")


        if selected_brand_for_model:
            df_filtered_by_brand = df_pagos_cristal[
                (df_pagos_cristal['marca_vehiculo'] == selected_brand_for_model) &
                (df_pagos_cristal['tipo_cristal'] == selected_tipo_cristal)
            ].copy()
            
            # Obtenemos la lista única de modelos para esa marca
            available_modelos = sorted(df_filtered_by_brand['modelo_vehiculo'].unique().tolist())
        else:
            available_modelos = []
        PLACEHOLDER_ALL_MODELS = "TODOS LOS MODELOS"
        options_with_all = [PLACEHOLDER_ALL_MODELS] + available_modelos


        st.markdown(f"###### :arrow_right: **Seleccionar modelo de {selected_brand_for_model} para análisis histórico:**")
        col1, col2 = st.columns([1, 3])
        with col1:
            if options_with_all:
                # Usar el primer modelo disponible como default (índice 0)
                selected_model_raw = st.selectbox(
                    "Modelo",
                    options=options_with_all,
                    index=0,
                    label_visibility='collapsed',
                    placeholder=PLACEHOLDER_ALL_MODELS,
                )
            else:
                st.info("No hay modelos disponibles para la marca seleccionada.")
                selected_model_raw = None

        if selected_model_raw:
            if selected_model_raw == PLACEHOLDER_ALL_MODELS:
                # Si se selecciona "TODOS LOS MODELOS", la lista de modelos a filtrar
                # debe ser la lista completa de modelos disponibles para esa marca.
                selected_model_list = available_modelos
                display_title = PLACEHOLDER_ALL_MODELS
            else:
                # Si se selecciona un modelo específico, la lista a filtrar es solo ese modelo.
                selected_model_list = [selected_model_raw]
                display_title = selected_model_raw

            df_historico = df_pagos_cristal[
                (df_pagos_cristal['modelo_vehiculo'].isin(selected_model_list)) &
                (df_pagos_cristal['marca_vehiculo'] == selected_brand_for_model) &
                (df_pagos_cristal['tipo_cristal'] == selected_tipo_cristal)
            ].copy()

            
            st.markdown(f"#### Histórico de Pagos: **{selected_brand_for_model} - {display_title}**")

            df_historico['Fecha Agrupación'] = pd.to_datetime(df_historico['año_mes_fecha_pago']).dt.strftime('%Y-%m')
            
            # df_historico['Año mes Fecha Pago'] = df_historico['año_mes_fecha_pago'] 
            
            agg_cols = {
                'monto_transaccion': 'mean', # Promedio de Monto Transaccion
                'pago_ipc': 'mean', # Promedio de Pago IPC
                'pago_usd': 'mean', # Promedio de Pago USD
            }
            
            df_tabla_final = df_historico.groupby('Fecha Agrupación').agg(agg_cols).reset_index()

            # 3.3. Renombrar las columnas para visualización
            df_tabla_final.rename(columns={
                'Fecha Agrupación': 'Año mes Fecha Pago',
                'monto_transaccion': 'Promedio de Monto Transaccion',
                'pago_ipc': 'Promedio de Pago IPC x Fecha de Pago',
                'pago_usd': 'Promedio de Pago USD x Fecha de Pago',
            }, inplace=True)

            df_tabla_final.sort_values(by='Año mes Fecha Pago', inplace=True)
            
            # 3.4. Agregar la fila de "Total Resultado" (Promedio General)
            
            promedio_general = df_tabla_final.mean(numeric_only=True)
            
            fila_total = pd.DataFrame([{
                'Año mes Fecha Pago': 'Total Resultado',
                'Promedio de Monto Transaccion': promedio_general['Promedio de Monto Transaccion'],
                'Promedio de Pago IPC x Fecha de Pago': promedio_general['Promedio de Pago IPC x Fecha de Pago'],
                'Promedio de Pago USD x Fecha de Pago': promedio_general['Promedio de Pago USD x Fecha de Pago'],
            }])
            
            df_tabla_final = pd.concat([df_tabla_final, fila_total], ignore_index=True)
            
            
            # Formatear la columna ARS/Total sin separador de miles (solo 0 decimales)
            df_tabla_final['Promedio de Monto Transaccion'] = df_tabla_final['Promedio de Monto Transaccion'].map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
            
            # Formatear las columnas USD/IPC (0 decimales)
            for col in ['Promedio de Pago IPC x Fecha de Pago', 'Promedio de Pago USD x Fecha de Pago']:
                df_tabla_final[col] = df_tabla_final[col].map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
            
            
            st.dataframe(df_tabla_final, use_container_width=True, hide_index=True, height=500, width=900)


            MONTO_COLS = ['monto_transaccion', 'pago_ipc', 'pago_usd']

            # Calculamos los promedios
            promedios_globales = df_pagos_ruedas[MONTO_COLS].mean().round(2)

            promedios_marca = df_filtered_by_brand[MONTO_COLS].mean().round(2)

            promedios_modelo = df_historico[MONTO_COLS].mean().round(2)

            diferencias_porcentuales = (
                (promedios_modelo - promedios_marca) / promedios_marca
            ) * 100
            diferencias_porcentuales = diferencias_porcentuales.round(1)

            st.markdown("#### :memo: Resumen de promedios y comparativa")

            data_resumen = {
                'Métrica': [
                    'Promedio de Pagos',
                    'Promedio de Pagos IPC',
                    'Promedio de Pagos USD'
                ],
                f'Todas las Marcas (General)': promedios_globales.tolist(),
                f'{selected_brand_for_model} (General)': promedios_marca.tolist(),
                f'{display_title} (Modelo)': promedios_modelo.tolist(),
                'Dif. Modelo vs Marca (%)': [
                    f"{diferencias_porcentuales['monto_transaccion']:.1f} %",
                    f"{diferencias_porcentuales['pago_ipc']:.1f} %",
                    f"{diferencias_porcentuales['pago_usd']:.1f} %",
                ]
            }

            df_resumen = pd.DataFrame(data_resumen)

            # Formatear las columnas de Promedio (Monetario, para mejor visualización)
            for col in [f'Todas las Marcas (General)', f'{selected_brand_for_model} (General)', f'{display_title} (Modelo)']:
                # Usamos f-string para formatear con separadores de miles y 0 decimales
                df_resumen[col] = df_resumen[col].apply(lambda x: f"{x:,.0f}") 

            st.dataframe(df_resumen, hide_index=True, use_container_width=True)



# ==========================================================================
# ---- Análisis PAGOS Robo de ruedas ---------------------------------------
# ==========================================================================


    elif current_analysis == opcion_6:
        st.markdown('## Evolución de monto de pagos de Robo de ruedas (L2)')    
        st.markdown("---")

        def create_plot_pagos(df_source, y1, y2, y3, title, x_tickangle=45):

            df_filtered = df_source[
                (df_source['marca_vehiculo'].isin(final_marcas_to_filter)) 
                & (df_source['tv'] ==selected_tv)
            ].copy()

            df_filtered.sort_values('año_mes_fecha_pago', inplace=True)
            df_plot = df_filtered.groupby('año_mes_fecha_pago').agg(
                {
                y1: 'mean',
                y2: 'mean',           
                y3: 'mean'         
                }).reset_index()

            # Columnas y etiquetas específicas para el gráfico (ajustadas al gráfico de la imagen)
            y1_cols = [y1, y2] # Eje ARS (Primario)
            y2_cols = [y3]                        # Eje USD (Secundario)
            
            y1_label = "Monto (ARS)"
            y2_label = "Monto (USD)"
            x_col = 'año_mes_fecha_pago'


            line_colors = {
                y1: '#1f77b4', 
                y2: '#ff7f0e',       
                y3: '#2ca02c'           
            }
            legend_names = {
                y1: y1,
                y2: y2,
                y3: y3
            }

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            #  eje primario
            for col in y1_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot[x_col], 
                        y=df_plot[col], 
                        name=legend_names[col],
                        line=dict(color=line_colors[col], width=3), 
                        showlegend=True,
                        # hovertemplate=f"{legend_names[col]}: %{{y:.2f}} ARS<extra></extra>"
                    ),
                    secondary_y=False, # Eje Y Izquierdo
                )

            # eje secundario
            for col in y2_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot[x_col], 
                        y=df_plot[col], 
                        name=legend_names[col],
                        line=dict(color=line_colors[col], width=3),
                        showlegend=True,
                        # hovertemplate=f"{legend_names[col]}: %{{y:.2f}} USD<extra></extra>"
                    ),
                    secondary_y=True, # Eje Y Derecho
                )


            fig.update_yaxes(title_text=f"{y1_label}", secondary_y=False, nticks=14, showgrid=True)
            fig.update_yaxes(title_text=f"{y2_label}", secondary_y=True, nticks=14, showgrid=False)

            # Ajustes del Gráfico
            fig.update_layout(
                title_text=title,
                height=700,
                legend_title_text='', # Dejar vacío ya que el nombre de la línea lo explica
                font=dict(family="Arial", size=15),
                margin=dict(t=100, b=0, l=0, r=0),
                title=dict(
                    font=dict(size=20, family="Arial"),
                ),
            )

            # eje X
            fig.update_xaxes(
                tickangle=x_tickangle, 
                showticklabels=True,
                title_text=''
            )
                    
            return fig
        
        def create_plot_pagos_marcas(df, y_col, color, facet_col, y_label, title, x_tickangle=None):

            if df.empty:
                fig = go.Figure().update_layout(
                    title_text=f"No hay datos para graficar",
                    height=400,
                    font=dict(family="Arial", size=10),
                    title_font_size=12
                )
                return fig

            df_filtered = df[
                (df['tv'] == selected_tv) &
                (df['marca_vehiculo'].isin(final_marcas_to_filter)) 
                # & (df['zona'].isin(selected_zone))
            ].copy()

            df_filtered.sort_values('año_mes_fecha_pago', inplace=True)
            df_plot = df_filtered.groupby(['año_mes_fecha_pago', 'marca_vehiculo']).agg(
                {
                'monto_transaccion': 'mean',
                'pago_usd': 'mean',           
                'pago_ipc': 'mean'         
                }).reset_index()
            
            fig = px.line(
                df_plot,
                x='año_mes_fecha_pago',
                y=y_col,
                color=color,
                color_discrete_sequence=["#FB0D0D", "lightgreen", "blue", "gray", "magenta", "cyan", "orange", '#2CA02C'],
                facet_col=facet_col,
                labels={'año_mes_fecha_pago': '', y_col: y_label,}# 'marca': 'Marca', 'tipo_cristal': 'Tipo de Cristal'}
            )

            fig.update_layout(
                title_text=title,
                height=700, # Altura del subplot individual
                # width=200,
                legend_title_text='',
                font=dict(family="Arial", size=15),
                margin=dict(t=50, b=0, l=0, r=0),
                title=dict(
                    font=dict(size=20, family="Arial")       
            ),
                legend=dict(
                orientation="h",        # Muestra la leyenda horizontalmente
                yanchor="top",          # Anclamos la leyenda en la parte superior del espacio que le damos (y)
                y=-0.2,                 # Colocamos la leyenda debajo del gráfico (ajusta este valor si es necesario)
                xanchor="center",       # Anclamos la leyenda en su centro
                x=0.5)                   # Posicionamos el centro de la leyenda en el medio del eje X (0.5)
                )

            fig.for_each_xaxis(
            lambda xaxis: xaxis.update(
                tickangle=x_tickangle, # Aplicar el ángulo deseado
                showticklabels=True     # Asegurarse de que las etiquetas sean visibles (si fuera necesario)
                )
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig.update_traces(line=dict(width=2))
            
            return fig
        
        available_marcas = sorted(df_pagos_ruedas['marca_vehiculo'].unique().tolist())
        available_marcas_todas = ["TODAS (general)"] + available_marcas
        available_tv = sorted(df_pagos_ruedas['tv'].unique().tolist())
        DEFAULT_MARCAS = ["VOLKSWAGEN", "CHEVROLET", "FORD",  "TOYOTA", "FIAT", "PEUGEOT", "RENAULT"]

        st.markdown("#### _Fuente de datos:_ \
            \n:white_small_square: _La Segunda BI (pagos)_")
        
        st.markdown('---')

        # selectbox para tipo de vehiculo
        with st.sidebar:
            st.markdown("---")
            st.markdown("Filtros")
            st.markdown("##### _Seleccionar Tipo de Vehículo:_") 
            selected_tv = st.selectbox(
                "TV",
                options=available_tv,
                index=0,
                label_visibility ='collapsed',
            )
            st.markdown("---")

            if selected_tv:
                df_filtered_by_tv = df_pagos_ruedas[df_pagos_ruedas['tv'] == selected_tv]
                available_marcas_for_tv = sorted(df_filtered_by_tv['marca_vehiculo'].unique().tolist())
            else:
                available_marcas_for_tv = sorted(df_pagos_ruedas['marca_vehiculo'].unique().tolist())

            default_selection = [
                m for m in DEFAULT_MARCAS 
                if m in available_marcas_for_tv
            ]

            # multiselect para diferentes marcas
            st.markdown("##### _Seleccionar Marcas:_")
            selected_marcas = st.multiselect(
                "Marcas",
                options=available_marcas_for_tv,
                default=default_selection,
                label_visibility='collapsed',
                placeholder="Seleccione una o varias Marcas..."
            )
            st.markdown("---")

            if not selected_marcas:
                final_marcas_to_filter = available_marcas_for_tv
                st.info(f"Filtro de Marcas: **TODAS** ({len(available_marcas_for_tv)} marcas)")
            else:
                final_marcas_to_filter = selected_marcas

# ----- PAGOS ROBO DE RUEDAS EVOLUTIVO --------------------------------------------------
        fig_pagos_ruedas = create_plot_pagos(
            df_pagos_ruedas, 
            'monto_transaccion',
            'pago_ipc',
            'pago_usd',
            title=f'Pagos promedio de ruedas (L2) - {selected_tv}', 
            x_tickangle=45
        )
        st.plotly_chart(fig_pagos_ruedas, use_container_width=True)
        
        df_pagos_ruedas.sort_values('año_mes_fecha_pago', inplace=True)
        pagos_ruedas_filtered = df_pagos_ruedas[
            (df_pagos_ruedas['marca_vehiculo'].isin(final_marcas_to_filter)) 
            & (df_pagos_ruedas['tv'] ==selected_tv)
        ]
        pagos_ruedas_filtered = pagos_ruedas_filtered.groupby(['año_mes_fecha_pago']).agg(
            {'monto_transaccion': 'mean',
            'pago_ipc': 'mean',           
            'pago_usd': 'mean'}).reset_index()
        
        pagos_ruedas_filtered.monto_transaccion = pagos_ruedas_filtered.monto_transaccion.round(0).astype(int) 
        pagos_ruedas_filtered.pago_usd = pagos_ruedas_filtered.pago_usd.round(0).astype(int) 
        pagos_ruedas_filtered.pago_ipc = pagos_ruedas_filtered.pago_ipc.round(0).astype(int)

        with st.expander("Ver tabla de datos (resumen)", icon=":material/query_stats:"):
            st.dataframe(pagos_ruedas_filtered, hide_index=True, width=900)

        st.subheader('', divider='grey')


# ----- ROBO DE RUEDAS POR MARCA --------------------------------------------------

        fig_pagos_ruedas_hist = create_plot_pagos_marcas(
            df_pagos_ruedas,
            'monto_transaccion', 
            'marca_vehiculo',
            None,
            'Monto histórico',
            title=f'Pagos robo de ruedas históricos por marca - {selected_tv}', 
            x_tickangle=45)
        
        fig_pagos_ruedas_ipc = create_plot_pagos_marcas(
            df_pagos_ruedas,
            'pago_ipc', 
            'marca_vehiculo',
            None,
            'Monto IPC',
            title=f'Pagos robo de ruedas por marca ajustados por IPC - {selected_tv}', 
            x_tickangle=45)
        
        fig_pagos_ruedas_usd = create_plot_pagos_marcas(
            df_pagos_ruedas,
            'pago_usd', 
            'marca_vehiculo',
            None,
            'Monto USD',
            title=f'Pagos robo de ruedas por marca en valor USD - {selected_tv}', 
            x_tickangle=45)


        tab1, tab2, tab3 = st.tabs(["Histórico", "IPC", 'USD'])
        with tab1:
            st.plotly_chart(fig_pagos_ruedas_hist, use_container_width=True)
        with tab2:
            st.plotly_chart(fig_pagos_ruedas_ipc, use_container_width=True)
        with tab3:
            st.plotly_chart(fig_pagos_ruedas_usd, use_container_width=True)
                    
        df_resumen_ruedas = df_pagos_ruedas.sort_values(by=['año_mes_fecha_pago', 'marca_vehiculo'])
        df_resumen_ruedas = df_resumen_ruedas[(df_resumen_ruedas['tv'] == selected_tv)
                & (df_resumen_ruedas['marca_vehiculo'].isin(final_marcas_to_filter))
            ]
        df_tabla_ruedas = df_resumen_ruedas.groupby(['año_mes_fecha_pago','marca_vehiculo']).agg(
            {'monto_transaccion': 'mean',
            'pago_usd': 'mean',           
            'pago_ipc': 'mean'}).reset_index()
        
        df_tabla_ruedas.monto_transaccion = df_tabla_ruedas.monto_transaccion.round(0).astype(int) 
        df_tabla_ruedas.pago_usd = df_tabla_ruedas.pago_usd.round(0).astype(int) 
        df_tabla_ruedas.pago_ipc = df_tabla_ruedas.pago_ipc.round(0).astype(int)

        with st.expander("Ver tabla de datos (resumen)", icon=":material/query_stats:"):
            st.dataframe(df_tabla_ruedas, hide_index=True, width=900)

        st.subheader('', divider='grey')

        
# ----- COMPARATIVO PARTICIPACION DE MERCADO ROBO DE RUEDAS POR MARCA --------------------------

        column_1, column_2 = st.columns(2)

        with column_1:
            st.markdown(f'#### :white_small_square: Participación de pagos por Marca - {selected_tv}')
            df_participacion = df_pagos_ruedas.sort_values(by=['marca_vehiculo'])
            df_participacion = df_participacion[
                (df_participacion['tv'] == selected_tv)
                ]
            df_participacion = df_participacion.groupby(['marca_vehiculo']).agg(
                {'monto_transaccion': 'sum',}).reset_index()
            
            monto_total = df_participacion['monto_transaccion'].sum()
            df_participacion['% Participación'] = (
                df_participacion['monto_transaccion'] / monto_total
            ) * 100
            
            df_participacion['% Participación'] = df_participacion['% Participación'].round(0).astype(int)
            
            df_participacion.sort_values(
                '% Participación', 
                ascending=False, 
                inplace=True
            )
            df_participacion['Part. Acum.'] = df_participacion['% Participación'].cumsum()
            df_participacion['Part. Acum.'] = df_participacion['Part. Acum.'].round(0).astype(int)


            def format_participacion(row): 

                participacion_individual_str = f"{row['% Participación']} %"
                
                # corte en 90% de acumulacion de pagos
                if row['Part. Acum.'] > 90.00:
                    # Si supera el 90%, el acumulado es vacío, pero el individual se mantiene
                    participacion_acumulada_str = ''
                else:
                    # Si es <= 90%, formateamos el acumulado
                    participacion_acumulada_str = f"{row['Part. Acum.']} %" 

                return pd.Series(
                    [participacion_individual_str, participacion_acumulada_str], 
                    index=['% Participación', 'Part. Acum. (%)']
                )

            df_participacion[['% Participación', 'Part. Acum. (%)']] = df_participacion.apply(
                format_participacion, 
                axis=1 
            )
            df_participacion.drop(columns=['Part. Acum.'], inplace=True)


            fila_total = pd.DataFrame([{
                'marca_vehiculo': 'TOTAL',
                'monto_transaccion': monto_total,
                '% Participación': '',
                'Part. Acum. (%)': ''
            }])

            # Concatenar la fila total al final del DataFrame
            df_participacion = pd.concat(
                [df_participacion, fila_total],
                ignore_index=True
            )

            st.dataframe(df_participacion.rename(columns={'monto_transaccion':'Suma de Pagos'}), hide_index=True, width=450, height=600)

# ----- PARTICIPACION POR MODELO --------------------------------------------------
        with column_2:
            st.markdown('#### :white_small_square: Participación de pagos por Modelo')

            TARGET_BRAND = "TOYOTA"

            if TARGET_BRAND in available_marcas_for_tv:
                # Si TOYOTA existe, encontramos su índice en la lista FINAL (que tiene el placeholder en la pos 0)
                default_index = available_marcas_for_tv.index(TARGET_BRAND)
            else:

                default_index = 0

            selected_brand_for_model = st.selectbox(
                "Seleccionar Marca:",
                options=available_marcas_for_tv,
                index=default_index,
                label_visibility='visible',
                placeholder="Seleccione una Marca...",
            )

            if selected_brand_for_model and selected_tv:
                # filtro por modelo
                df_modelos = df_pagos_ruedas[
                    (df_pagos_ruedas['marca_vehiculo'] == selected_brand_for_model) &
                    (df_pagos_ruedas['tv'] == selected_tv)
                ].copy()
                
                df_modelos_participacion = df_modelos.groupby('modelo_vehiculo').agg(
                    {'monto_transaccion': 'sum'}
                ).reset_index()
                
                df_modelos_participacion.rename(
                    columns={'monto_transaccion': 'Suma de Pagos'}, 
                    inplace=True
                )

                df_modelos_participacion.sort_values(
                    'Suma de Pagos', 
                    ascending=False, 
                    inplace=True
                )
                monto_total_marca = df_modelos_participacion['Suma de Pagos'].sum()

                df_modelos_participacion['% Participación'] = (
                    df_modelos_participacion['Suma de Pagos'] / monto_total_marca
                ) * 100
                
                # Redondear y formatear el porcentaje
                df_modelos_participacion['% Participación'] = (
                    df_modelos_participacion['% Participación']
                    .round(0)
                    .astype(int)
                    .astype(str) + ' %'
                )

                st.dataframe(df_modelos_participacion, hide_index=True, width=450, height=600)

            else:
                st.info("Por favor, seleccione una marca de vehículo para ver la participación por modelo.")


        if selected_brand_for_model:
            df_filtered_by_brand = df_pagos_ruedas[
                (df_pagos_ruedas['marca_vehiculo'] == selected_brand_for_model) &
                (df_pagos_ruedas['tv'] == selected_tv)
            ].copy()
            
            # Obtenemos la lista única de modelos para esa marca
            available_modelos = sorted(df_filtered_by_brand['modelo_vehiculo'].unique().tolist())
        else:
            available_modelos = []
        PLACEHOLDER_ALL_MODELS = "TODOS LOS MODELOS"
        options_with_all = [PLACEHOLDER_ALL_MODELS] + available_modelos


        st.markdown(f"###### :arrow_right: **Seleccionar modelo de {selected_brand_for_model} para análisis histórico:**")
        col1, col2 = st.columns([1, 3])
        with col1:
            if options_with_all:
                # Usar el primer modelo disponible como default (índice 0)
                selected_model_raw = st.selectbox(
                    "Modelo",
                    options=options_with_all,
                    index=0,
                    label_visibility='collapsed',
                    placeholder=PLACEHOLDER_ALL_MODELS,
                )
            else:
                st.info("No hay modelos disponibles para la marca seleccionada.")
                selected_model_raw = None

        if selected_model_raw:
            if selected_model_raw == PLACEHOLDER_ALL_MODELS:
                # Si se selecciona "TODOS LOS MODELOS", la lista de modelos a filtrar
                # debe ser la lista completa de modelos disponibles para esa marca.
                selected_model_list = available_modelos
                display_title = PLACEHOLDER_ALL_MODELS
            else:
                # Si se selecciona un modelo específico, la lista a filtrar es solo ese modelo.
                selected_model_list = [selected_model_raw]
                display_title = selected_model_raw

            df_historico = df_pagos_ruedas[
                (df_pagos_ruedas['modelo_vehiculo'].isin(selected_model_list)) &
                (df_pagos_ruedas['marca_vehiculo'] == selected_brand_for_model) &
                (df_pagos_ruedas['tv'] == selected_tv)
            ].copy()

            
            st.markdown(f"#### Histórico de Pagos: **{selected_brand_for_model} - {display_title}**")

            df_historico['Fecha Agrupación'] = pd.to_datetime(df_historico['año_mes_fecha_pago']).dt.strftime('%Y-%m')
            
            # df_historico['Año mes Fecha Pago'] = df_historico['año_mes_fecha_pago'] 
            
            agg_cols = {
                'monto_transaccion': 'mean', # Promedio de Monto Transaccion
                'pago_ipc': 'mean', # Promedio de Pago IPC
                'pago_usd': 'mean', # Promedio de Pago USD
            }
            
            df_tabla_final = df_historico.groupby('Fecha Agrupación').agg(agg_cols).reset_index()

            # 3.3. Renombrar las columnas para visualización
            df_tabla_final.rename(columns={
                'Fecha Agrupación': 'Año mes Fecha Pago',
                'monto_transaccion': 'Promedio de Monto Transaccion',
                'pago_ipc': 'Promedio de Pago IPC x Fecha de Pago',
                'pago_usd': 'Promedio de Pago USD x Fecha de Pago',
            }, inplace=True)

            df_tabla_final.sort_values(by='Año mes Fecha Pago', inplace=True)
            
            # 3.4. Agregar la fila de "Total Resultado" (Promedio General)
            
            promedio_general = df_tabla_final.mean(numeric_only=True)
            
            fila_total = pd.DataFrame([{
                'Año mes Fecha Pago': 'Total Resultado',
                'Promedio de Monto Transaccion': promedio_general['Promedio de Monto Transaccion'],
                'Promedio de Pago IPC x Fecha de Pago': promedio_general['Promedio de Pago IPC x Fecha de Pago'],
                'Promedio de Pago USD x Fecha de Pago': promedio_general['Promedio de Pago USD x Fecha de Pago'],
            }])
            
            df_tabla_final = pd.concat([df_tabla_final, fila_total], ignore_index=True)
            
            
            # Formatear la columna ARS/Total sin separador de miles (solo 0 decimales)
            df_tabla_final['Promedio de Monto Transaccion'] = df_tabla_final['Promedio de Monto Transaccion'].map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
            
            # Formatear las columnas USD/IPC (0 decimales)
            for col in ['Promedio de Pago IPC x Fecha de Pago', 'Promedio de Pago USD x Fecha de Pago']:
                df_tabla_final[col] = df_tabla_final[col].map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
            
            
            st.dataframe(df_tabla_final, use_container_width=True, hide_index=True, height=500, width=900)


            MONTO_COLS = ['monto_transaccion', 'pago_ipc', 'pago_usd']

            # Calculamos los promedios
            promedios_globales = df_pagos_ruedas[MONTO_COLS].mean().round(2)

            promedios_marca = df_filtered_by_brand[MONTO_COLS].mean().round(2)

            promedios_modelo = df_historico[MONTO_COLS].mean().round(2)

            diferencias_porcentuales = (
                (promedios_modelo - promedios_marca) / promedios_marca
            ) * 100
            diferencias_porcentuales = diferencias_porcentuales.round(1)

            st.markdown("#### :memo: Resumen de promedios y comparativa")

            data_resumen = {
                'Métrica': [
                    'Promedio de Pagos',
                    'Promedio de Pagos IPC',
                    'Promedio de Pagos USD'
                ],
                f'Todas las Marcas (General)': promedios_globales.tolist(),
                f'{selected_brand_for_model} (General)': promedios_marca.tolist(),
                f'{display_title} (Modelo)': promedios_modelo.tolist(),
                'Dif. Modelo vs Marca (%)': [
                    f"{diferencias_porcentuales['monto_transaccion']:.1f} %",
                    f"{diferencias_porcentuales['pago_ipc']:.1f} %",
                    f"{diferencias_porcentuales['pago_usd']:.1f} %",
                ]
            }

            df_resumen = pd.DataFrame(data_resumen)

            # Formatear las columnas de Promedio (Monetario, para mejor visualización)
            for col in [f'Todas las Marcas (General)', f'{selected_brand_for_model} (General)', f'{display_title} (Modelo)']:
                # Usamos f-string para formatear con separadores de miles y 0 decimales
                df_resumen[col] = df_resumen[col].apply(lambda x: f"{x:,.0f}") 

            st.dataframe(df_resumen, hide_index=True, use_container_width=True)

# =====================================================================================
# --- Evolución de monto de pagos de Daños Materiales 
# =====================================================================================

    elif current_analysis == opcion_7:
            st.markdown('## Evolución de monto de pagos de Daños Materiales (L2)')    
            st.markdown("---")

            def create_plot_pagos(df_source, y1, y2, y3, title, x_tickangle=45):

                df_filtered = df_source[
                    (df_source['marca_vehiculo'].isin(final_marcas_to_filter)) 
                    & (df_source['tv'] ==selected_tv)
                ].copy()

                df_filtered.sort_values('año_mes_fecha_pago', inplace=True)
                df_plot = df_filtered.groupby('año_mes_fecha_pago').agg(
                    {
                    y1: 'mean',
                    y2: 'mean',           
                    y3: 'mean'         
                    }).reset_index()

                # Columnas y etiquetas específicas para el gráfico (ajustadas al gráfico de la imagen)
                y1_cols = [y1, y2] # Eje ARS (Primario)
                y2_cols = [y3]                        # Eje USD (Secundario)
                
                y1_label = "Monto (ARS)"
                y2_label = "Monto (USD)"
                x_col = 'año_mes_fecha_pago'


                line_colors = {
                    y1: '#1f77b4', 
                    y2: '#ff7f0e',       
                    y3: '#2ca02c'           
                }
                legend_names = {
                    y1: y1,
                    y2: y2,
                    y3: y3
                }

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                #  eje primario
                for col in y1_cols:
                    fig.add_trace(
                        go.Scatter(
                            x=df_plot[x_col], 
                            y=df_plot[col], 
                            name=legend_names[col],
                            line=dict(color=line_colors[col], width=3), 
                            showlegend=True,
                            # hovertemplate=f"{legend_names[col]}: %{{y:.2f}} ARS<extra></extra>"
                        ),
                        secondary_y=False, # Eje Y Izquierdo
                    )

                # eje secundario
                for col in y2_cols:
                    fig.add_trace(
                        go.Scatter(
                            x=df_plot[x_col], 
                            y=df_plot[col], 
                            name=legend_names[col],
                            line=dict(color=line_colors[col], width=3),
                            showlegend=True,
                            # hovertemplate=f"{legend_names[col]}: %{{y:.2f}} USD<extra></extra>"
                        ),
                        secondary_y=True, # Eje Y Derecho
                    )


                fig.update_yaxes(title_text=f"{y1_label}", secondary_y=False, nticks=14, showgrid=True)
                fig.update_yaxes(title_text=f"{y2_label}", secondary_y=True, nticks=14, showgrid=False)

                # Ajustes del Gráfico
                fig.update_layout(
                    title_text=title,
                    height=700,
                    legend_title_text='', # Dejar vacío ya que el nombre de la línea lo explica
                    font=dict(family="Arial", size=15),
                    margin=dict(t=100, b=0, l=0, r=0),
                    title=dict(
                        font=dict(size=20, family="Arial"),
                    ),
                )

                # eje X
                fig.update_xaxes(
                    tickangle=x_tickangle, 
                    showticklabels=True,
                    title_text=''
                )
                        
                return fig
            
            def create_plot_pagos_marcas(df, y_col, color, facet_col, y_label, title, x_tickangle=None):

                if df.empty:
                    fig = go.Figure().update_layout(
                        title_text=f"No hay datos para graficar",
                        height=400,
                        font=dict(family="Arial", size=10),
                        title_font_size=12
                    )
                    return fig

                df_filtered = df[
                    (df['tv'] == selected_tv) &
                    (df['marca_vehiculo'].isin(final_marcas_to_filter)) 
                    # & (df['zona'].isin(selected_zone))
                ].copy()

                df_filtered.sort_values('año_mes_fecha_pago', inplace=True)
                df_plot = df_filtered.groupby(['año_mes_fecha_pago', 'marca_vehiculo']).agg(
                    {
                    'monto_transaccion': 'mean',
                    'pago_usd': 'mean',           
                    'pago_ipc': 'mean'         
                    }).reset_index()
                
                fig = px.line(
                    df_plot,
                    x='año_mes_fecha_pago',
                    y=y_col,
                    color=color,
                    color_discrete_sequence=["#FB0D0D", "lightgreen", "blue", "gray", "magenta", "cyan", "orange", '#2CA02C'],
                    facet_col=facet_col,
                    labels={'año_mes_fecha_pago': '', y_col: y_label,}# 'marca': 'Marca', 'tipo_cristal': 'Tipo de Cristal'}
                )

                fig.update_layout(
                    title_text=title,
                    height=700, # Altura del subplot individual
                    # width=200,
                    legend_title_text='',
                    font=dict(family="Arial", size=15),
                    margin=dict(t=50, b=0, l=0, r=0),
                    title=dict(
                        font=dict(size=20, family="Arial")       
                ),
                    legend=dict(
                    orientation="h",        # Muestra la leyenda horizontalmente
                    yanchor="top",          # Anclamos la leyenda en la parte superior del espacio que le damos (y)
                    y=-0.2,                 # Colocamos la leyenda debajo del gráfico (ajusta este valor si es necesario)
                    xanchor="center",       # Anclamos la leyenda en su centro
                    x=0.5)                   # Posicionamos el centro de la leyenda en el medio del eje X (0.5)
                    )

                fig.for_each_xaxis(
                lambda xaxis: xaxis.update(
                    tickangle=x_tickangle, # Aplicar el ángulo deseado
                    showticklabels=True     # Asegurarse de que las etiquetas sean visibles (si fuera necesario)
                    )
                )
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                fig.update_traces(line=dict(width=2))
                
                return fig
            
            available_marcas = sorted(df_pagos_materiales['marca_vehiculo'].unique().tolist())
            available_marcas_todas = ["TODAS (general)"] + available_marcas
            available_tv = sorted(df_pagos_materiales['tv'].unique().tolist())
            DEFAULT_MARCAS = ["VOLKSWAGEN", "CHEVROLET", "FORD",  "TOYOTA", "FIAT", "PEUGEOT", "RENAULT"]

            st.markdown("#### _Fuente de datos:_ \
                \n:white_small_square: _La Segunda BI (pagos)_")
            
            st.markdown('---')

            # selectbox para tipo de vehiculo
            with st.sidebar:
                st.markdown("---")
                st.markdown("Filtros")
                st.markdown("##### _Seleccionar Tipo de Vehículo:_") 
                selected_tv = st.selectbox(
                    "TV",
                    options=available_tv,
                    index=5,
                    label_visibility ='collapsed',
                )
                st.markdown("---")

                if selected_tv:
                    df_filtered_by_tv = df_pagos_materiales[df_pagos_materiales['tv'] == selected_tv]
                    available_marcas_for_tv = sorted(df_filtered_by_tv['marca_vehiculo'].unique().tolist())
                else:
                    available_marcas_for_tv = sorted(df_pagos_materiales['marca_vehiculo'].unique().tolist())

                default_selection = [
                    m for m in DEFAULT_MARCAS 
                    if m in available_marcas_for_tv
                ]

                # multiselect para diferentes marcas
                st.markdown("##### _Seleccionar Marcas:_")
                selected_marcas = st.multiselect(
                    "Marcas",
                    options=available_marcas_for_tv,
                    default=default_selection,
                    label_visibility='collapsed',
                    placeholder="Seleccione una o varias Marcas..."
                )
                st.markdown("---")

                if not selected_marcas:
                    final_marcas_to_filter = available_marcas_for_tv
                    st.info(f"Filtro de Marcas: **TODAS** ({len(available_marcas_for_tv)} marcas)")
                else:
                    final_marcas_to_filter = selected_marcas

    # ----- PAGOS ROBO DE RUEDAS EVOLUTIVO --------------------------------------------------
            agg_cols = {
                'monto_transaccion': 'sum', # Promedio de Monto Transaccion
                'pago_ipc': 'sum', # Promedio de Pago IPC
                'pago_usd': 'sum', # Promedio de Pago USD
            }

            tabla = df_pagos_materiales.groupby('tv').agg(agg_cols).reset_index().sort_values(by='monto_transaccion',ascending=False).rename(columns={'tv':'cobertura'})

            st.markdown("###### :memo: Resumen de pagos por TV")
            
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                st.dataframe(tabla, use_container_width=True, hide_index=True)

            fig_pagos_mat = create_plot_pagos(
                df_pagos_materiales, 
                'monto_transaccion',
                'pago_ipc',
                'pago_usd',
                title=f'Pagos promedio de daños materiales (L2) - {selected_tv}', 
                x_tickangle=45
            )
            st.plotly_chart(fig_pagos_mat, use_container_width=True)
            
            df_pagos_materiales.sort_values('año_mes_fecha_pago', inplace=True)
            pagos_mat_filtered = df_pagos_materiales[
                (df_pagos_materiales['marca_vehiculo'].isin(final_marcas_to_filter)) 
                & (df_pagos_materiales['tv'] ==selected_tv)
            ]
            pagos_mat_filtered = pagos_mat_filtered.groupby(['año_mes_fecha_pago']).agg(
                {'monto_transaccion': 'mean',
                'pago_ipc': 'mean',           
                'pago_usd': 'mean'}).reset_index()
            
            pagos_mat_filtered.monto_transaccion = pagos_mat_filtered.monto_transaccion.round(0).astype(int) 
            pagos_mat_filtered.pago_usd = pagos_mat_filtered.pago_usd.round(0).astype(int) 
            pagos_mat_filtered.pago_ipc = pagos_mat_filtered.pago_ipc.round(0).astype(int)

            with st.expander("Ver tabla de datos (resumen)", icon=":material/query_stats:"):
                st.dataframe(pagos_mat_filtered, hide_index=True, width=900)

            st.subheader('', divider='grey')


    # ----- ROBO DE RUEDAS POR MARCA --------------------------------------------------

            fig_pagos_mat_hist = create_plot_pagos_marcas(
                df_pagos_materiales,
                'monto_transaccion', 
                'marca_vehiculo',
                None,
                'Monto histórico',
                title=f'Pagos robo de ruedas históricos por marca - {selected_tv}', 
                x_tickangle=45)
            
            fig_pagos_mat_ipc = create_plot_pagos_marcas(
                df_pagos_materiales,
                'pago_ipc', 
                'marca_vehiculo',
                None,
                'Monto IPC',
                title=f'Pagos robo de ruedas por marca ajustados por IPC - {selected_tv}', 
                x_tickangle=45)
            
            fig_pagos_mat_usd = create_plot_pagos_marcas(
                df_pagos_materiales,
                'pago_usd', 
                'marca_vehiculo',
                None,
                'Monto USD',
                title=f'Pagos robo de ruedas por marca en valor USD - {selected_tv}', 
                x_tickangle=45)


            tab1, tab2, tab3 = st.tabs(["Histórico", "IPC", 'USD'])
            with tab1:
                st.plotly_chart(fig_pagos_mat_hist, use_container_width=True)
            with tab2:
                st.plotly_chart(fig_pagos_mat_ipc, use_container_width=True)
            with tab3:
                st.plotly_chart(fig_pagos_mat_usd, use_container_width=True)
                        
            df_resumen_mat = df_pagos_materiales.sort_values(by=['año_mes_fecha_pago', 'marca_vehiculo'])
            df_resumen_mat = df_resumen_mat[(df_resumen_mat['tv'] == selected_tv)
                    & (df_resumen_mat['marca_vehiculo'].isin(final_marcas_to_filter))
                ]
            df_tabla_mat = df_resumen_mat.groupby(['año_mes_fecha_pago','marca_vehiculo']).agg(
                {'monto_transaccion': 'mean',
                'pago_usd': 'mean',           
                'pago_ipc': 'mean'}).reset_index()
            
            df_tabla_mat.monto_transaccion = df_tabla_mat.monto_transaccion.round(0).astype(int) 
            df_tabla_mat.pago_usd = df_tabla_mat.pago_usd.round(0).astype(int) 
            df_tabla_mat.pago_ipc = df_tabla_mat.pago_ipc.round(0).astype(int)

            with st.expander("Ver tabla de datos (resumen)", icon=":material/query_stats:"):
                st.dataframe(df_tabla_mat, hide_index=True, width=900)

            st.subheader('', divider='grey')

            
    # ----- COMPARATIVO PARTICIPACION DE MERCADO ROBO DE RUEDAS POR MARCA --------------------------

            column_1, column_2 = st.columns(2)

            with column_1:
                st.markdown(f'#### :white_small_square: Participación de pagos por Marca - {selected_tv}')
                df_participacion = df_pagos_materiales.sort_values(by=['marca_vehiculo'])
                df_participacion = df_participacion[
                    (df_participacion['tv'] == selected_tv)
                    ]
                df_participacion = df_participacion.groupby(['marca_vehiculo']).agg(
                    {'monto_transaccion': 'sum',}).reset_index()
                
                monto_total = df_participacion['monto_transaccion'].sum()
                df_participacion['% Participación'] = (
                    df_participacion['monto_transaccion'] / monto_total
                ) * 100
                
                df_participacion['% Participación'] = df_participacion['% Participación'].round(0).astype(int)
                
                df_participacion.sort_values(
                    '% Participación', 
                    ascending=False, 
                    inplace=True
                )
                df_participacion['Part. Acum.'] = df_participacion['% Participación'].cumsum()
                df_participacion['Part. Acum.'] = df_participacion['Part. Acum.'].round(0).astype(int)


                def format_participacion(row): 

                    participacion_individual_str = f"{row['% Participación']} %"
                    
                    # corte en 90% de acumulacion de pagos
                    if row['Part. Acum.'] > 90.00:
                        # Si supera el 90%, el acumulado es vacío, pero el individual se mantiene
                        participacion_acumulada_str = ''
                    else:
                        # Si es <= 90%, formateamos el acumulado
                        participacion_acumulada_str = f"{row['Part. Acum.']} %" 

                    return pd.Series(
                        [participacion_individual_str, participacion_acumulada_str], 
                        index=['% Participación', 'Part. Acum. (%)']
                    )

                df_participacion[['% Participación', 'Part. Acum. (%)']] = df_participacion.apply(
                    format_participacion, 
                    axis=1 
                )
                df_participacion.drop(columns=['Part. Acum.'], inplace=True)


                fila_total = pd.DataFrame([{
                    'marca_vehiculo': 'TOTAL',
                    'monto_transaccion': monto_total,
                    '% Participación': '',
                    'Part. Acum. (%)': ''
                }])

                # Concatenar la fila total al final del DataFrame
                df_participacion = pd.concat(
                    [df_participacion, fila_total],
                    ignore_index=True
                )

                st.dataframe(df_participacion.rename(columns={'monto_transaccion':'Suma de Pagos'}), hide_index=True, width=450, height=600)

    # ----- PARTICIPACION POR MODELO --------------------------------------------------
            with column_2:
                st.markdown('#### :white_small_square: Participación de pagos por Modelo')

                TARGET_BRAND = "TOYOTA"

                if TARGET_BRAND in available_marcas_for_tv:
                    # Si TOYOTA existe, encontramos su índice en la lista FINAL (que tiene el placeholder en la pos 0)
                    default_index = available_marcas_for_tv.index(TARGET_BRAND)
                else:

                    default_index = 0

                selected_brand_for_model = st.selectbox(
                    "Seleccionar Marca:",
                    options=available_marcas_for_tv,
                    index=default_index,
                    label_visibility='visible',
                    placeholder="Seleccione una Marca...",
                )

                if selected_brand_for_model and selected_tv:
                    # filtro por modelo
                    df_modelos = df_pagos_materiales[
                        (df_pagos_materiales['marca_vehiculo'] == selected_brand_for_model) &
                        (df_pagos_materiales['tv'] == selected_tv)
                    ].copy()
                    
                    df_modelos_participacion = df_modelos.groupby('modelo_vehiculo').agg(
                        {'monto_transaccion': 'sum'}
                    ).reset_index()
                    
                    df_modelos_participacion.rename(
                        columns={'monto_transaccion': 'Suma de Pagos'}, 
                        inplace=True
                    )

                    df_modelos_participacion.sort_values(
                        'Suma de Pagos', 
                        ascending=False, 
                        inplace=True
                    )
                    monto_total_marca = df_modelos_participacion['Suma de Pagos'].sum()

                    df_modelos_participacion['% Participación'] = (
                        df_modelos_participacion['Suma de Pagos'] / monto_total_marca
                    ) * 100
                    
                    # Redondear y formatear el porcentaje
                    df_modelos_participacion['% Participación'] = (
                        df_modelos_participacion['% Participación']
                        .round(0)
                        .astype(int)
                        .astype(str) + ' %'
                    )

                    st.dataframe(df_modelos_participacion, hide_index=True, width=450, height=600)

                else:
                    st.info("Por favor, seleccione una marca de vehículo para ver la participación por modelo.")


            if selected_brand_for_model:
                df_filtered_by_brand = df_pagos_materiales[
                    (df_pagos_materiales['marca_vehiculo'] == selected_brand_for_model) &
                    (df_pagos_materiales['tv'] == selected_tv)
                ].copy()
                
                # Obtenemos la lista única de modelos para esa marca
                available_modelos = sorted(df_filtered_by_brand['modelo_vehiculo'].unique().tolist())
            else:
                available_modelos = []
            PLACEHOLDER_ALL_MODELS = "TODOS LOS MODELOS"
            options_with_all = [PLACEHOLDER_ALL_MODELS] + available_modelos


            st.markdown(f"###### :arrow_right: **Seleccionar modelo de {selected_brand_for_model} para análisis histórico:**")
            col1, col2 = st.columns([1, 3])
            with col1:
                if options_with_all:
                    # Usar el primer modelo disponible como default (índice 0)
                    selected_model_raw = st.selectbox(
                        "Modelo",
                        options=options_with_all,
                        index=0,
                        label_visibility='collapsed',
                        placeholder=PLACEHOLDER_ALL_MODELS,
                    )
                else:
                    st.info("No hay modelos disponibles para la marca seleccionada.")
                    selected_model_raw = None

            if selected_model_raw:
                if selected_model_raw == PLACEHOLDER_ALL_MODELS:
                    # Si se selecciona "TODOS LOS MODELOS", la lista de modelos a filtrar
                    # debe ser la lista completa de modelos disponibles para esa marca.
                    selected_model_list = available_modelos
                    display_title = PLACEHOLDER_ALL_MODELS
                else:
                    # Si se selecciona un modelo específico, la lista a filtrar es solo ese modelo.
                    selected_model_list = [selected_model_raw]
                    display_title = selected_model_raw

                df_historico = df_pagos_materiales[
                    (df_pagos_materiales['modelo_vehiculo'].isin(selected_model_list)) &
                    (df_pagos_materiales['marca_vehiculo'] == selected_brand_for_model) &
                    (df_pagos_materiales['tv'] == selected_tv)
                ].copy()

                
                st.markdown(f"#### Histórico de Pagos: **{selected_brand_for_model} - {display_title}**")

                df_historico['Fecha Agrupación'] = pd.to_datetime(df_historico['año_mes_fecha_pago']).dt.strftime('%Y-%m')
                
                # df_historico['Año mes Fecha Pago'] = df_historico['año_mes_fecha_pago'] 
                
                agg_cols = {
                    'monto_transaccion': 'mean', # Promedio de Monto Transaccion
                    'pago_ipc': 'mean', # Promedio de Pago IPC
                    'pago_usd': 'mean', # Promedio de Pago USD
                }
                
                df_tabla_final = df_historico.groupby('Fecha Agrupación').agg(agg_cols).reset_index()

                # 3.3. Renombrar las columnas para visualización
                df_tabla_final.rename(columns={
                    'Fecha Agrupación': 'Año mes Fecha Pago',
                    'monto_transaccion': 'Promedio de Monto Transaccion',
                    'pago_ipc': 'Promedio de Pago IPC x Fecha de Pago',
                    'pago_usd': 'Promedio de Pago USD x Fecha de Pago',
                }, inplace=True)

                df_tabla_final.sort_values(by='Año mes Fecha Pago', inplace=True)
                
                # 3.4. Agregar la fila de "Total Resultado" (Promedio General)
                
                promedio_general = df_tabla_final.mean(numeric_only=True)
                
                fila_total = pd.DataFrame([{
                    'Año mes Fecha Pago': 'Total Resultado',
                    'Promedio de Monto Transaccion': promedio_general['Promedio de Monto Transaccion'],
                    'Promedio de Pago IPC x Fecha de Pago': promedio_general['Promedio de Pago IPC x Fecha de Pago'],
                    'Promedio de Pago USD x Fecha de Pago': promedio_general['Promedio de Pago USD x Fecha de Pago'],
                }])
                
                df_tabla_final = pd.concat([df_tabla_final, fila_total], ignore_index=True)
                
                
                # Formatear la columna ARS/Total sin separador de miles (solo 0 decimales)
                df_tabla_final['Promedio de Monto Transaccion'] = df_tabla_final['Promedio de Monto Transaccion'].map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
                
                # Formatear las columnas USD/IPC (0 decimales)
                for col in ['Promedio de Pago IPC x Fecha de Pago', 'Promedio de Pago USD x Fecha de Pago']:
                    df_tabla_final[col] = df_tabla_final[col].map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
                
                
                st.dataframe(df_tabla_final, use_container_width=True, hide_index=True, height=500, width=900)


                MONTO_COLS = ['monto_transaccion', 'pago_ipc', 'pago_usd']

                # Calculamos los promedios
                promedios_globales = df_pagos_materiales[MONTO_COLS].mean().round(2)

                promedios_marca = df_filtered_by_brand[MONTO_COLS].mean().round(2)

                promedios_modelo = df_historico[MONTO_COLS].mean().round(2)

                diferencias_porcentuales = (
                    (promedios_modelo - promedios_marca) / promedios_marca
                ) * 100
                diferencias_porcentuales = diferencias_porcentuales.round(1)

                st.markdown("#### :memo: Resumen de promedios y comparativa")

                data_resumen = {
                    'Métrica': [
                        'Promedio de Pagos',
                        'Promedio de Pagos IPC',
                        'Promedio de Pagos USD'
                    ],
                    f'Todas las Marcas (General)': promedios_globales.tolist(),
                    f'{selected_brand_for_model} (General)': promedios_marca.tolist(),
                    f'{display_title} (Modelo)': promedios_modelo.tolist(),
                    'Dif. vs Marca (%)': [
                        f"{diferencias_porcentuales['monto_transaccion']:.1f} %",
                        f"{diferencias_porcentuales['pago_ipc']:.1f} %",
                        f"{diferencias_porcentuales['pago_usd']:.1f} %",
                    ]
                }

                df_resumen = pd.DataFrame(data_resumen)

                # Formatear las columnas de Promedio (Monetario, para mejor visualización)
                for col in [f'Todas las Marcas (General)', f'{selected_brand_for_model} (General)', f'{display_title} (Modelo)']:
                    # Usamos f-string para formatear con separadores de miles y 0 decimales
                    df_resumen[col] = df_resumen[col].apply(lambda x: f"{x:,.0f}") 

                st.dataframe(df_resumen, hide_index=True, use_container_width=True)


# --- PAGOS CASCOS -----------------------------------------------------

    elif current_analysis == opcion_8:
                st.markdown('## Evolución de monto de pagos de Cascos (L2)')    
                st.markdown("---")

                def create_plot_pagos(df_source, y1, y2, y3, title, x_tickangle=45):

                    df_filtered = df_source[
                        (df_source['marca_vehiculo'].isin(final_marcas_to_filter)) 
                        & (df_source['cobertura_principal'] ==selected_tv)
                    ].copy()

                    df_filtered.sort_values('año_mes_fecha_pago', inplace=True)
                    df_plot = df_filtered.groupby('año_mes_fecha_pago').agg(
                        {
                        y1: 'mean',
                        y2: 'mean',           
                        y3: 'mean'         
                        }).reset_index()

                    # Columnas y etiquetas específicas para el gráfico (ajustadas al gráfico de la imagen)
                    y1_cols = [y1, y2] # Eje ARS (Primario)
                    y2_cols = [y3]                        # Eje USD (Secundario)
                    
                    y1_label = "Monto (ARS)"
                    y2_label = "Monto (USD)"
                    x_col = 'año_mes_fecha_pago'


                    line_colors = {
                        y1: '#1f77b4', 
                        y2: '#ff7f0e',       
                        y3: '#2ca02c'           
                    }
                    legend_names = {
                        y1: y1,
                        y2: y2,
                        y3: y3
                    }

                    fig = make_subplots(specs=[[{"secondary_y": True}]])

                    #  eje primario
                    for col in y1_cols:
                        fig.add_trace(
                            go.Scatter(
                                x=df_plot[x_col], 
                                y=df_plot[col], 
                                name=legend_names[col],
                                line=dict(color=line_colors[col], width=3), 
                                showlegend=True,
                                # hovertemplate=f"{legend_names[col]}: %{{y:.2f}} ARS<extra></extra>"
                            ),
                            secondary_y=False, # Eje Y Izquierdo
                        )

                    # eje secundario
                    for col in y2_cols:
                        fig.add_trace(
                            go.Scatter(
                                x=df_plot[x_col], 
                                y=df_plot[col], 
                                name=legend_names[col],
                                line=dict(color=line_colors[col], width=3),
                                showlegend=True,
                                # hovertemplate=f"{legend_names[col]}: %{{y:.2f}} USD<extra></extra>"
                            ),
                            secondary_y=True, # Eje Y Derecho
                        )


                    fig.update_yaxes(title_text=f"{y1_label}", secondary_y=False, nticks=14, showgrid=True)
                    fig.update_yaxes(title_text=f"{y2_label}", secondary_y=True, nticks=14, showgrid=False)

                    # Ajustes del Gráfico
                    fig.update_layout(
                        title_text=title,
                        height=700,
                        legend_title_text='', # Dejar vacío ya que el nombre de la línea lo explica
                        font=dict(family="Arial", size=15),
                        margin=dict(t=100, b=0, l=0, r=0),
                        title=dict(
                            font=dict(size=20, family="Arial"),
                        ),
                    )

                    # eje X
                    fig.update_xaxes(
                        tickangle=x_tickangle, 
                        showticklabels=True,
                        title_text=''
                    )
                            
                    return fig
                
                def create_plot_pagos_marcas(df, y_col, color, facet_col, y_label, title, x_tickangle=None):

                    if df.empty:
                        fig = go.Figure().update_layout(
                            title_text=f"No hay datos para graficar",
                            height=400,
                            font=dict(family="Arial", size=10),
                            title_font_size=12
                        )
                        return fig

                    df_filtered = df[
                        (df['cobertura_principal'] == selected_tv) &
                        (df['marca_vehiculo'].isin(final_marcas_to_filter)) 
                        # & (df['zona'].isin(selected_zone))
                    ].copy()

                    df_filtered.sort_values('año_mes_fecha_pago', inplace=True)
                    df_plot = df_filtered.groupby(['año_mes_fecha_pago', 'marca_vehiculo']).agg(
                        {
                        'monto_transaccion': 'mean',
                        'pago_usd': 'mean',           
                        'pago_ipc': 'mean'         
                        }).reset_index()
                    
                    fig = px.line(
                        df_plot,
                        x='año_mes_fecha_pago',
                        y=y_col,
                        color=color,
                        color_discrete_sequence=["#FB0D0D", "lightgreen", "blue", "gray", "magenta", "cyan", "orange", '#2CA02C'],
                        facet_col=facet_col,
                        labels={'año_mes_fecha_pago': '', y_col: y_label,}# 'marca': 'Marca', 'tipo_cristal': 'Tipo de Cristal'}
                    )

                    fig.update_layout(
                        title_text=title,
                        height=700, # Altura del subplot individual
                        # width=200,
                        legend_title_text='',
                        font=dict(family="Arial", size=15),
                        margin=dict(t=50, b=0, l=0, r=0),
                        title=dict(
                            font=dict(size=20, family="Arial")       
                    ),
                        legend=dict(
                        orientation="h",        # Muestra la leyenda horizontalmente
                        yanchor="top",          # Anclamos la leyenda en la parte superior del espacio que le damos (y)
                        y=-0.2,                 # Colocamos la leyenda debajo del gráfico (ajusta este valor si es necesario)
                        xanchor="center",       # Anclamos la leyenda en su centro
                        x=0.5)                   # Posicionamos el centro de la leyenda en el medio del eje X (0.5)
                        )

                    fig.for_each_xaxis(
                    lambda xaxis: xaxis.update(
                        tickangle=x_tickangle, # Aplicar el ángulo deseado
                        showticklabels=True     # Asegurarse de que las etiquetas sean visibles (si fuera necesario)
                        )
                    )
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                    fig.update_traces(line=dict(width=2))
                    
                    return fig
                
                available_marcas = sorted(df_pagos_cascos['marca_vehiculo'].unique().tolist())
                available_marcas_todas = ["TODAS (general)"] + available_marcas
                available_tv = sorted(df_pagos_cascos['cobertura_principal'].unique().tolist())
                DEFAULT_MARCAS = ["VOLKSWAGEN", "CHEVROLET", "FORD",  "TOYOTA", "FIAT", "PEUGEOT", "RENAULT"]

                st.markdown("#### _Fuente de datos:_ \
                    \n:white_small_square: _La Segunda BI (pagos)_")
                
                st.markdown('---')

                # selectbox para tipo de vehiculo
                with st.sidebar:
                    st.markdown("---")
                    st.markdown("Filtros")
                    st.markdown("##### _Seleccionar Cobertura:_") 
                    selected_tv = st.selectbox(
                        "TV",
                        options=available_tv,
                        index=0,
                        label_visibility ='collapsed',
                    )
                    st.markdown("---")

                    if selected_tv:
                        df_filtered_by_tv = df_pagos_cascos[df_pagos_cascos['cobertura_principal'] == selected_tv]
                        available_marcas_for_tv = sorted(df_filtered_by_tv['marca_vehiculo'].unique().tolist())
                    else:
                        available_marcas_for_tv = sorted(df_pagos_cascos['marca_vehiculo'].unique().tolist())

                    default_selection = [
                        m for m in DEFAULT_MARCAS 
                        if m in available_marcas_for_tv
                    ]

                    # multiselect para diferentes marcas
                    st.markdown("##### _Seleccionar Marcas:_")
                    selected_marcas = st.multiselect(
                        "Marcas",
                        options=available_marcas_for_tv,
                        default=default_selection,
                        label_visibility='collapsed',
                        placeholder="Seleccione una o varias Marcas..."
                    )
                    st.markdown("---")

                    if not selected_marcas:
                        final_marcas_to_filter = available_marcas_for_tv
                        st.info(f"Filtro de Marcas: **TODAS** ({len(available_marcas_for_tv)} marcas)")
                    else:
                        final_marcas_to_filter = selected_marcas

        # ----- PAGOS ROBO DE RUEDAS EVOLUTIVO --------------------------------------------------
                agg_cols = {
                    'monto_transaccion': 'sum', # Promedio de Monto Transaccion
                    'pago_ipc': 'sum', # Promedio de Pago IPC
                    'pago_usd': 'sum', # Promedio de Pago USD
                }

                tabla = df_pagos_cascos.groupby('cobertura_principal').agg(agg_cols).reset_index().sort_values(by='monto_transaccion',ascending=False).rename(columns={'cobertura_principal':'cobertura'})

                st.markdown("###### :memo: Resumen de pagos por TV")
                
                with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                    st.dataframe(tabla, use_container_width=True, hide_index=True)

                fig_pagos_mat = create_plot_pagos(
                    df_pagos_cascos, 
                    'monto_transaccion',
                    'pago_ipc',
                    'pago_usd',
                    title=f'Pagos promedio de daños materiales (L2) - {selected_tv}', 
                    x_tickangle=45
                )
                st.plotly_chart(fig_pagos_mat, use_container_width=True)
                
                df_pagos_cascos.sort_values('año_mes_fecha_pago', inplace=True)
                pagos_mat_filtered = df_pagos_cascos[
                    (df_pagos_cascos['marca_vehiculo'].isin(final_marcas_to_filter)) 
                    & (df_pagos_cascos['cobertura_principal'] ==selected_tv)
                ]
                pagos_mat_filtered = pagos_mat_filtered.groupby(['año_mes_fecha_pago']).agg(
                    {'monto_transaccion': 'mean',
                    'pago_ipc': 'mean',           
                    'pago_usd': 'mean'}).reset_index()
                
                pagos_mat_filtered.monto_transaccion = pagos_mat_filtered.monto_transaccion.round(0).astype(int) 
                pagos_mat_filtered.pago_usd = pagos_mat_filtered.pago_usd.round(0).astype(int) 
                pagos_mat_filtered.pago_ipc = pagos_mat_filtered.pago_ipc.round(0).astype(int)

                with st.expander("Ver tabla de datos (resumen)", icon=":material/query_stats:"):
                    st.dataframe(pagos_mat_filtered, hide_index=True, width=900)

                st.subheader('', divider='grey')


        # ----- ROBO DE RUEDAS POR MARCA --------------------------------------------------

                fig_pagos_mat_hist = create_plot_pagos_marcas(
                    df_pagos_cascos,
                    'monto_transaccion', 
                    'marca_vehiculo',
                    None,
                    'Monto histórico',
                    title=f'Pagos robo de ruedas históricos por marca - {selected_tv}', 
                    x_tickangle=45)
                
                fig_pagos_mat_ipc = create_plot_pagos_marcas(
                    df_pagos_cascos,
                    'pago_ipc', 
                    'marca_vehiculo',
                    None,
                    'Monto IPC',
                    title=f'Pagos robo de ruedas por marca ajustados por IPC - {selected_tv}', 
                    x_tickangle=45)
                
                fig_pagos_mat_usd = create_plot_pagos_marcas(
                    df_pagos_cascos,
                    'pago_usd', 
                    'marca_vehiculo',
                    None,
                    'Monto USD',
                    title=f'Pagos robo de ruedas por marca en valor USD - {selected_tv}', 
                    x_tickangle=45)


                tab1, tab2, tab3 = st.tabs(["Histórico", "IPC", 'USD'])
                with tab1:
                    st.plotly_chart(fig_pagos_mat_hist, use_container_width=True)
                with tab2:
                    st.plotly_chart(fig_pagos_mat_ipc, use_container_width=True)
                with tab3:
                    st.plotly_chart(fig_pagos_mat_usd, use_container_width=True)
                            
                df_resumen_mat = df_pagos_cascos.sort_values(by=['año_mes_fecha_pago', 'marca_vehiculo'])
                df_resumen_mat = df_resumen_mat[(df_resumen_mat['cobertura_principal'] == selected_tv)
                        & (df_resumen_mat['marca_vehiculo'].isin(final_marcas_to_filter))
                    ]
                df_tabla_mat = df_resumen_mat.groupby(['año_mes_fecha_pago','marca_vehiculo']).agg(
                    {'monto_transaccion': 'mean',
                    'pago_usd': 'mean',           
                    'pago_ipc': 'mean'}).reset_index()
                
                df_tabla_mat.monto_transaccion = df_tabla_mat.monto_transaccion.round(0).astype(int) 
                df_tabla_mat.pago_usd = df_tabla_mat.pago_usd.round(0).astype(int) 
                df_tabla_mat.pago_ipc = df_tabla_mat.pago_ipc.round(0).astype(int)

                with st.expander("Ver tabla de datos (resumen)", icon=":material/query_stats:"):
                    st.dataframe(df_tabla_mat, hide_index=True, width=900)

                st.subheader('', divider='grey')

                
        # ----- COMPARATIVO PARTICIPACION DE MERCADO ROBO DE RUEDAS POR MARCA --------------------------

                column_1, column_2 = st.columns(2)

                with column_1:
                    st.markdown(f'#### :white_small_square: Participación de pagos por Marca - {selected_tv}')
                    df_participacion = df_pagos_cascos.sort_values(by=['marca_vehiculo'])
                    df_participacion = df_participacion[
                        (df_participacion['cobertura_principal'] == selected_tv)
                        ]
                    df_participacion = df_participacion.groupby(['marca_vehiculo']).agg(
                        {'monto_transaccion': 'sum',}).reset_index()
                    
                    monto_total = df_participacion['monto_transaccion'].sum()
                    df_participacion['% Participación'] = (
                        df_participacion['monto_transaccion'] / monto_total
                    ) * 100
                    
                    df_participacion['% Participación'] = df_participacion['% Participación'].round(0).astype(int)
                    
                    df_participacion.sort_values(
                        '% Participación', 
                        ascending=False, 
                        inplace=True
                    )
                    df_participacion['Part. Acum.'] = df_participacion['% Participación'].cumsum()
                    df_participacion['Part. Acum.'] = df_participacion['Part. Acum.'].round(0).astype(int)


                    def format_participacion(row): 

                        participacion_individual_str = f"{row['% Participación']} %"
                        
                        # corte en 90% de acumulacion de pagos
                        if row['Part. Acum.'] > 90.00:
                            # Si supera el 90%, el acumulado es vacío, pero el individual se mantiene
                            participacion_acumulada_str = ''
                        else:
                            # Si es <= 90%, formateamos el acumulado
                            participacion_acumulada_str = f"{row['Part. Acum.']} %" 

                        return pd.Series(
                            [participacion_individual_str, participacion_acumulada_str], 
                            index=['% Participación', 'Part. Acum. (%)']
                        )

                    df_participacion[['% Participación', 'Part. Acum. (%)']] = df_participacion.apply(
                        format_participacion, 
                        axis=1 
                    )
                    df_participacion.drop(columns=['Part. Acum.'], inplace=True)


                    fila_total = pd.DataFrame([{
                        'marca_vehiculo': 'TOTAL',
                        'monto_transaccion': monto_total,
                        '% Participación': '',
                        'Part. Acum. (%)': ''
                    }])

                    # Concatenar la fila total al final del DataFrame
                    df_participacion = pd.concat(
                        [df_participacion, fila_total],
                        ignore_index=True
                    )

                    st.dataframe(df_participacion.rename(columns={'monto_transaccion':'Suma de Pagos'}), hide_index=True, width=450, height=600)

        # ----- PARTICIPACION POR MODELO --------------------------------------------------
                with column_2:
                    st.markdown('#### :white_small_square: Participación de pagos por Modelo')

                    TARGET_BRAND = "TOYOTA"

                    if TARGET_BRAND in available_marcas_for_tv:
                        # Si TOYOTA existe, encontramos su índice en la lista FINAL (que tiene el placeholder en la pos 0)
                        default_index = available_marcas_for_tv.index(TARGET_BRAND)
                    else:

                        default_index = 0

                    selected_brand_for_model = st.selectbox(
                        "Seleccionar Marca:",
                        options=available_marcas_for_tv,
                        index=default_index,
                        label_visibility='visible',
                        placeholder="Seleccione una Marca...",
                    )

                    if selected_brand_for_model and selected_tv:
                        # filtro por modelo
                        df_modelos = df_pagos_cascos[
                            (df_pagos_cascos['marca_vehiculo'] == selected_brand_for_model) &
                            (df_pagos_cascos['cobertura_principal'] == selected_tv)
                        ].copy()
                        
                        df_modelos_participacion = df_modelos.groupby('modelo_vehiculo').agg(
                            {'monto_transaccion': 'sum'}
                        ).reset_index()
                        
                        df_modelos_participacion.rename(
                            columns={'monto_transaccion': 'Suma de Pagos'}, 
                            inplace=True
                        )

                        df_modelos_participacion.sort_values(
                            'Suma de Pagos', 
                            ascending=False, 
                            inplace=True
                        )
                        monto_total_marca = df_modelos_participacion['Suma de Pagos'].sum()

                        df_modelos_participacion['% Participación'] = (
                            df_modelos_participacion['Suma de Pagos'] / monto_total_marca
                        ) * 100
                        
                        # Redondear y formatear el porcentaje
                        df_modelos_participacion['% Participación'] = (
                            df_modelos_participacion['% Participación']
                            .round(0)
                            .astype(int)
                            .astype(str) + ' %'
                        )

                        st.dataframe(df_modelos_participacion, hide_index=True, width=450, height=600)

                    else:
                        st.info("Por favor, seleccione una marca de vehículo para ver la participación por modelo.")


                if selected_brand_for_model:
                    df_filtered_by_brand = df_pagos_cascos[
                        (df_pagos_cascos['marca_vehiculo'] == selected_brand_for_model) &
                        (df_pagos_cascos['cobertura_principal'] == selected_tv)
                    ].copy()
                    
                    # Obtenemos la lista única de modelos para esa marca
                    available_modelos = sorted(df_filtered_by_brand['modelo_vehiculo'].unique().tolist())
                else:
                    available_modelos = []
                PLACEHOLDER_ALL_MODELS = "TODOS LOS MODELOS"
                options_with_all = [PLACEHOLDER_ALL_MODELS] + available_modelos


                st.markdown(f"###### :arrow_right: **Seleccionar modelo de {selected_brand_for_model} para análisis histórico:**")
                col1, col2 = st.columns([1, 3])
                with col1:
                    if options_with_all:
                        # Usar el primer modelo disponible como default (índice 0)
                        selected_model_raw = st.selectbox(
                            "Modelo",
                            options=options_with_all,
                            index=0,
                            label_visibility='collapsed',
                            placeholder=PLACEHOLDER_ALL_MODELS,
                        )
                    else:
                        st.info("No hay modelos disponibles para la marca seleccionada.")
                        selected_model_raw = None

                if selected_model_raw:
                    if selected_model_raw == PLACEHOLDER_ALL_MODELS:
                        # Si se selecciona "TODOS LOS MODELOS", la lista de modelos a filtrar
                        # debe ser la lista completa de modelos disponibles para esa marca.
                        selected_model_list = available_modelos
                        display_title = PLACEHOLDER_ALL_MODELS
                    else:
                        # Si se selecciona un modelo específico, la lista a filtrar es solo ese modelo.
                        selected_model_list = [selected_model_raw]
                        display_title = selected_model_raw

                    df_historico = df_pagos_cascos[
                        (df_pagos_cascos['modelo_vehiculo'].isin(selected_model_list)) &
                        (df_pagos_cascos['marca_vehiculo'] == selected_brand_for_model) &
                        (df_pagos_cascos['cobertura_principal'] == selected_tv)
                    ].copy()

                    
                    st.markdown(f"#### Histórico de Pagos: **{selected_brand_for_model} - {display_title}**")

                    df_historico['Fecha Agrupación'] = pd.to_datetime(df_historico['año_mes_fecha_pago']).dt.strftime('%Y-%m')
                    
                    # df_historico['Año mes Fecha Pago'] = df_historico['año_mes_fecha_pago'] 
                    
                    agg_cols = {
                        'monto_transaccion': 'mean', # Promedio de Monto Transaccion
                        'pago_ipc': 'mean', # Promedio de Pago IPC
                        'pago_usd': 'mean', # Promedio de Pago USD
                    }
                    
                    df_tabla_final = df_historico.groupby('Fecha Agrupación').agg(agg_cols).reset_index()

                    # 3.3. Renombrar las columnas para visualización
                    df_tabla_final.rename(columns={
                        'Fecha Agrupación': 'Año mes Fecha Pago',
                        'monto_transaccion': 'Promedio de Monto Transaccion',
                        'pago_ipc': 'Promedio de Pago IPC x Fecha de Pago',
                        'pago_usd': 'Promedio de Pago USD x Fecha de Pago',
                    }, inplace=True)

                    df_tabla_final.sort_values(by='Año mes Fecha Pago', inplace=True)
                    
                    # 3.4. Agregar la fila de "Total Resultado" (Promedio General)
                    
                    promedio_general = df_tabla_final.mean(numeric_only=True)
                    
                    fila_total = pd.DataFrame([{
                        'Año mes Fecha Pago': 'Total Resultado',
                        'Promedio de Monto Transaccion': promedio_general['Promedio de Monto Transaccion'],
                        'Promedio de Pago IPC x Fecha de Pago': promedio_general['Promedio de Pago IPC x Fecha de Pago'],
                        'Promedio de Pago USD x Fecha de Pago': promedio_general['Promedio de Pago USD x Fecha de Pago'],
                    }])
                    
                    df_tabla_final = pd.concat([df_tabla_final, fila_total], ignore_index=True)
                    
                    
                    # Formatear la columna ARS/Total sin separador de miles (solo 0 decimales)
                    df_tabla_final['Promedio de Monto Transaccion'] = df_tabla_final['Promedio de Monto Transaccion'].map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
                    
                    # Formatear las columnas USD/IPC (0 decimales)
                    for col in ['Promedio de Pago IPC x Fecha de Pago', 'Promedio de Pago USD x Fecha de Pago']:
                        df_tabla_final[col] = df_tabla_final[col].map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
                    
                    
                    st.dataframe(df_tabla_final, use_container_width=True, hide_index=True, height=500, width=900)


                    MONTO_COLS = ['monto_transaccion', 'pago_ipc', 'pago_usd']

                    # Calculamos los promedios
                    promedios_globales = df_pagos_cascos[MONTO_COLS].mean().round(2)

                    promedios_marca = df_filtered_by_brand[MONTO_COLS].mean().round(2)

                    promedios_modelo = df_historico[MONTO_COLS].mean().round(2)

                    diferencias_porcentuales = (
                        (promedios_modelo - promedios_marca) / promedios_marca
                    ) * 100
                    diferencias_porcentuales = diferencias_porcentuales.round(1)

                    st.markdown("#### :memo: Resumen de promedios y comparativa")

                    data_resumen = {
                        'Métrica': [
                            'Promedio de Pagos',
                            'Promedio de Pagos IPC',
                            'Promedio de Pagos USD'
                        ],
                        f'Todas las Marcas (General)': promedios_globales.tolist(),
                        f'{selected_brand_for_model} (General)': promedios_marca.tolist(),
                        f'{display_title} (Modelo)': promedios_modelo.tolist(),
                        'Dif. vs Marca (%)': [
                            f"{diferencias_porcentuales['monto_transaccion']:.1f} %",
                            f"{diferencias_porcentuales['pago_ipc']:.1f} %",
                            f"{diferencias_porcentuales['pago_usd']:.1f} %",
                        ]
                    }

                    df_resumen = pd.DataFrame(data_resumen)

                    # Formatear las columnas de Promedio (Monetario, para mejor visualización)
                    for col in [f'Todas las Marcas (General)', f'{selected_brand_for_model} (General)', f'{display_title} (Modelo)']:
                        # Usamos f-string para formatear con separadores de miles y 0 decimales
                        df_resumen[col] = df_resumen[col].apply(lambda x: f"{x:,.0f}") 

                    st.dataframe(df_resumen, hide_index=True, use_container_width=True)


    elif current_analysis == opcion_0:
        
        def generar_grafico_comparativo(df_long):
            
            fig = px.bar(
                df_long,
                x='Variable',
                y='Valor',
                color='marca',
                barmode='group',
                title="Comparativa por Marca (Participación y Ratios)",
                labels={'Valor': 'Valor (%)', 'Variable': 'Métrica Analizada', 'marca': 'Marca'},
                height=700,
                # width=1000,
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            # Ajustes estéticos
            fig.update_layout(
                showlegend=True,
                legend_title_text='',
                font=dict(family="Arial", size=11),
                margin=dict(t=80, b=50, l=50, r=50),
                hovermode="x unified",
                legend=dict(
                    orientation="h",        # leyenda horizontalmente
                    yanchor="top",          
                    y=-0.2, 
                    xanchor="center",  
                    x=0.5),
                bargap=0.15,    
                bargroupgap=0.1,
                title=dict(
                    font=dict(size=17, family="Arial")               
            ))
            fig.update_xaxes(title_text=None)
            # Añadimos etiquetas sobre las barras para mayor claridad
            fig.update_traces(
                texttemplate='%{y:.1f}', 
                textposition='outside',
                cliponaxis=False # Evita que se corten los números arriba
            )
            
            return fig
        

        st.markdown('### Composición de cartera - La Segunda')
        # fuente de datos bi
        st.markdown("#### _Fuente de datos:_ \
            \n:white_small_square: _La Segunda BI_")

        st.markdown('---') 

        st.markdown('##### :calendar: Período 2024-2025')
        st.dataframe(
            tabla_marcas_año,
            use_container_width=True,
            hide_index=True,)
        
        st.markdown('')
        st.markdown('##### :arrow_right: Composición de la cartera por Marca - Corte en 20 primeras marcas')
        st.dataframe(
            tabla_marcas_head_20[['marca', 'prima_dev_hist', 'prima_%','prima_%_acum', 'prima_dev_ipc','prima_dev_ipc_%', 'prima_dev_ipc_%_acum',
                'ar_t', 'ar_%', 'ar_%_acum', 'ns_t', 'ns_t_%', 'ns_t_%_acum',
                'ns_t_nro_sin', 'is_t', 'is_t_%', 'is_t_%_acum', 'is_t_ipc', 'is_t_ipc_%', 'is_t_ipc_%_acum',
                'sin', 'sin_ipc', 'frec']],
            use_container_width=True,
            hide_index=True,)
        
        st.markdown('')
        st.markdown('##### :arrow_right: Composición de la cartera por Marca - Corte en 8 primeras marcas')
        st.dataframe(
            tabla_marcas_head_8[['marca', 'prima_dev_hist', 'prima_%','prima_%_acum', 'prima_dev_ipc','prima_dev_ipc_%', 'prima_dev_ipc_%_acum',
                'ar_t', 'ar_%', 'ar_%_acum', 'ns_t', 'ns_t_%', 'ns_t_%_acum',
                'ns_t_nro_sin', 'is_t', 'is_t_%', 'is_t_%_acum', 'is_t_ipc', 'is_t_ipc_%', 'is_t_ipc_%_acum',
                'sin', 'sin_ipc', 'frec']],
            use_container_width=True,
            hide_index=True,)
        
        st.markdown('')
        # st.markdown('##### :: Gráfico comparativo de métricas clave por Marca')
        fig_comparativo = generar_grafico_comparativo(df_graf_cartera)
        st.plotly_chart(fig_comparativo, use_container_width=True)

# =======================================================================================
# --- COMPARATIVO SA vs COSTE MEDIO DE REPUESTOS ========================================
# =======================================================================================

    elif current_analysis == opcion_9:

        st.markdown("## Comparativa: Variación SA (InfoAuto) vs Coste medio de repuestos")
        st.markdown("#### _Período: marzo - diciembre 2025_")
        st.markdown("---")

        df_sa_rep['año_mes'] = pd.to_datetime(df_sa_rep['año_mes'])

        def generar_grafico_comparativo(df, col_var_rep, col_var_sa, titulo):

            fig = go.Figure()

            # Línea de Variación Repuestos
            fig.add_trace(go.Scatter(
                x=df['año_mes'], 
                y=df[col_var_rep],
                name="Var. Repuestos (%)",
                line=dict(color='#e2e8f0', width=3), # Color secundario
                mode='lines+markers'
            ))

            # Línea de Variación Suma Asegurada
            fig.add_trace(go.Scatter(
                x=df['año_mes'], 
                y=df[col_var_sa],
                name="Var. Suma Asegurada (%)",
                # hovertemplate='<b>%{data.name}</b>: %{y:.2f}%<extra></extra>',
                line=dict(color='#615fff', width=3), # Color primario
                mode='lines+markers'
            ))
            fig.update_traces(
                hovertemplate='<b>%{fullData.name}</b>: %{y:.2f}%<extra></extra>')

            # Línea de referencia en 0
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            fig.update_layout(
                title=titulo,
                xaxis_title="",
                yaxis_title="Variación Mensual (%)",
                hovermode="x unified",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=80, b=20, l=20, r=20),
                yaxis=dict(tickformat=".2f", ticksuffix="%")
            )
            return fig
        

        def generar_grafico_comparativo_facetado(df, col_var_rep, col_var_sa, titulo, col_ref=None):
            """
            Genera un gráfico comparativo facetado por marca para dos variables de variación.
            """
            df_plot = df.copy()
            
            columnas_interes = [col_var_rep, col_var_sa]
            if col_ref:
                columnas_interes.append(col_ref)

            df_melted = df_plot.melt(
                id_vars=['año_mes', 'marca'],
                value_vars=columnas_interes,
                var_name='Métrica',
                value_name='Variación'
            )

            nombres_map = {
                col_var_rep: 'Var. Repuestos (%)',
                col_var_sa: 'Var. Suma Asegurada (%)'
            }
    
            if col_ref:
                nombres_map[col_ref] = 'IPC'
            
            df_melted['Métrica'] = df_melted['Métrica'].map(nombres_map)

            # 3. Construcción del Gráfico con Plotly Express
            # Usamos facet_col='marca' para crear la grilla
            fig = px.line(
                df_melted,
                x='año_mes',
                y='Variación',
                color='Métrica',
                facet_col='marca',
                facet_col_wrap=2,
                markers=True,
                color_discrete_map={
                    'Var. Suma Asegurada (%)': '#615fff', 
                    'Var. Repuestos (%)': '#e2e8f0',
                    'IPC': '#ff6361'
                },
                # line_dash_map={
                #     'Var. Suma Asegurada (%)': 'solid',
                #     'Var. Repuestos (%)': 'solid',
                #     'IPC': 'dot'
                # },
                labels={'año_mes': '', 'Variación': 'Variación Mensual (%)'}
            )

            fig.for_each_trace(lambda t: t.update(
                line=dict(dash='dot') if 'IPC' in t.name else dict(dash='solid'),
                mode='lines' if 'IPC' in t.name else 'lines+markers' # Quitamos marcadores al IPC para que se vea el punteado
            ))
            fig.update_traces(
                hovertemplate='<b>%{fullData.name}</b>: %{y:.2f}%<extra></extra>'
            )

            # Línea de referencia en 0 para cada subgráfico
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            # Configuración del Layout
            fig.update_layout(
                title=dict(text=f"<b>{titulo}</b>", x=0.01, font=dict(size=18)),
                xaxis_title="",
                hovermode="x unified",
                height=1000, # Altura para acomodar las marcas
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, title_text=None),
                margin=dict(t=100, b=40, l=40, r=40),
                font=dict(family="Arial")
            )

            # Formateo de Ejes Y (2 decimales y signo %)
            fig.update_yaxes(tickformat=".2f", ticksuffix="%", showgrid=True)
            fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))

            return fig
        
        
        with st.container(border=True):
            st.subheader("1. Comparativo histórico")
            fig_hist = generar_grafico_comparativo(
                df_sa_rep, 'var_costo_prom', 'var_sa', 
                "Variación Mensual: Repuestos vs SA (Nominal)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with st.container(border=True):
            st.subheader("2. Comparativa Ajustada por IPC")
            # st.info("Muestra si las variables ganaron o perdieron contra la inflación en términos reales.")
            fig_ipc = generar_grafico_comparativo(
                df_sa_rep, 'var_costo_prom_ipc', 'var_sa_ipc', 
                "Variación Mensual Real (Ajustado por IPC)"
            )
            st.plotly_chart(fig_ipc, use_container_width=True)

        with st.container(border=True):
            st.subheader("3. Comparativa en Dólar (USD Blue)")
            fig_usd = generar_grafico_comparativo(
                df_sa_rep, 'var_costo_prom_usd', 'var_sa_prom_usd', 
                "Variación Mensual en Moneda Dura (USD)"
            )
            st.plotly_chart(fig_usd, use_container_width=True)

        def calcular_resumen(df, col_rep, col_sa, label):
            # Usamos suma simple de variaciones para una aproximación rápida del periodo
            brecha = df[col_rep].sum() - df[col_sa].sum()
            if brecha > 0:
                return f"**{label}**: Los repuestos crecieron **{brecha:.1f} pts** más que la SA."
            else:
                return f"**{label}**: La SA creció **{abs(brecha):.1f} pts** más que los repuestos."
        
        st.markdown('')
        st.markdown("#### Data Cruda")

        df_sa_rep_raw = df_sa_rep[['año_mes', 'ipc', 'usd_blue',
                                    'costo_pieza_prom_hist','sa_prom', 'var_costo_prom','var_sa', 
                                    'costo_prom_ipc','sa_prom_ipc','var_costo_prom_ipc','var_sa_ipc',  
                                    'costo_prom_usd', 'sa_prom_usd', 'var_costo_prom_usd',
                                    'var_sa_prom_usd']]  

        st.dataframe(df_sa_rep_raw, 
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                    'año_mes': st.column_config.DateColumn("Año-Mes", format="YYYY-MM"),
                    'ipc': 'IPC',
                    'usd_blue': 'Valor USD blue',
                    "costo_pieza_prom_hist": st.column_config.NumberColumn("Coste medio rep. hist.", format="$ %.0f"),
                    "sa_prom": st.column_config.NumberColumn("SA prom.", format="$ %.0f"),
                    "var_costo_prom": st.column_config.NumberColumn(
                        "Var. costo rep. %", 
                        format="%.2f%%",
                    ),
                    "var_sa": st.column_config.NumberColumn(
                        "Var. SA %", 
                        format="%.2f%%",
                        ),
                    "costo_prom_ipc": st.column_config.NumberColumn("Coste medio rep. IPC", format="$ %.0f"),
                    "sa_prom_ipc": st.column_config.NumberColumn("SA prom. IPC", format="$ %.0f"),
                    "var_costo_prom_ipc": st.column_config.NumberColumn(                            
                        "Var. Costo rep. IPC %", 
                        format="%.2f%%",
                        ),
                    "var_sa_ipc": st.column_config.NumberColumn(
                        "Var. SA IPC %", 
                        format="%.2f%%",
                        ),
                    "costo_prom_usd": st.column_config.NumberColumn("Coste medio rep. USD", format="$ %.0f"),
                    "sa_prom_usd": st.column_config.NumberColumn("SA prom. USD", format="$ %.0f"),
                    "var_costo_prom_usd": st.column_config.NumberColumn(
                        "Var. Costo rep. USD %", 
                        format="%.2f%%",
                        ),
                    "var_sa_prom_usd": st.column_config.NumberColumn(
                        "Var. SA USD %", 
                        format="%.2f%%",
                        ),
                    })    
        
        st.markdown("---")
        st.markdown("## Variación SA (cartera L2) vs Coste medio de repuestos (Orion/Cesvi)")
        st.markdown("#### _Período: enero 2023 - diciembre 2025_")


        with st.container(border=True):

            st.markdown("#### 1. Comparativo: Variación SA (todos los planes) vs Costo medio repuestos")
            df_sa_rep_cartera_todos = df_sa_rep_cartera[df_sa_rep_cartera.plan=='Todos']
            fig_hist_cart = generar_grafico_comparativo(
                df_sa_rep_cartera_todos, 'var_costo_prom_hist', 'var_sa_prom', 
                "Variación histórica: Costo prom. repuestos vs SA prom."
            )
            fig_ipc_cart = generar_grafico_comparativo(
                df_sa_rep_cartera_todos, 'var_costo_prom_ipc', 'var_sa_prom_ipc', 
                "Variación IPC: Costo prom. repuestos vs SA prom."
            )
            fig_usd_cart = generar_grafico_comparativo(
                df_sa_rep_cartera_todos, 'var_costo_prom_usd', 'var_sa_prom_usd', 
                "Variación USD: Costo prom. repuestos vs SA prom."
            )
            
            fig_hist_cart.add_trace(go.Scatter(
                x=df_sa_rep_cartera_todos['año_mes'],
                y=df_sa_rep_cartera_todos['var_ipc'],
                name='IPC',        # Nombre que aparecerá en la leyenda
                mode='lines',
                line=dict(color='#ff6361', dash='dot')
            ))
            
            tab1, tab2, tab3 = st.tabs(["📊 Histórico", "📈 IPC", "💵 USD"])
            with tab1:
                st.plotly_chart(fig_hist_cart, use_container_width=True)
            with tab2:
                st.plotly_chart(fig_ipc_cart, use_container_width=True)
            with tab3:
                st.plotly_chart(fig_usd_cart, use_container_width=True)


        with st.container(border=True):

            st.markdown("#### 2. Comparativo: Variación SA (Plan 30) vs Costo medio repuestos")
            df_sa_rep_cartera_plan_30 = df_sa_rep_cartera[df_sa_rep_cartera.plan=='Plan 30']
            fig_hist_cart_plan30 = generar_grafico_comparativo(
                df_sa_rep_cartera_plan_30, 'var_costo_prom_hist', 'var_sa_prom', 
                "Variación histórica: Costo prom. repuestos vs SA prom."
            )
            fig_ipc_cart_plan30 = generar_grafico_comparativo(
                df_sa_rep_cartera_plan_30, 'var_costo_prom_ipc', 'var_sa_prom_ipc', 
                "Variación IPC: Costo prom. repuestos vs SA prom."
            )
            fig_usd_cart_plan30 = generar_grafico_comparativo(
                df_sa_rep_cartera_plan_30, 'var_costo_prom_usd', 'var_sa_prom_usd', 
                "Variación USD: Costo prom. repuestos vs SA prom."
            )
            
            fig_hist_cart_plan30.add_trace(go.Scatter(
                x=df_sa_rep_cartera_plan_30['año_mes'],
                y=df_sa_rep_cartera_plan_30['var_ipc'],
                name='IPC',        # Nombre que aparecerá en la leyenda
                mode='lines',
                line=dict(color='#ff6361', dash='dot')
            ))
            
            tab1, tab2, tab3 = st.tabs(["📊 Histórico", "📈 IPC", "💵 USD"])
            with tab1:
                st.plotly_chart(fig_hist_cart_plan30, use_container_width=True)
            with tab2:
                st.plotly_chart(fig_ipc_cart_plan30, use_container_width=True)
            with tab3:
                st.plotly_chart(fig_usd_cart_plan30, use_container_width=True)


        st.markdown('')
        st.markdown("#### Data Cruda")
        st.dataframe(df_sa_rep_cartera.drop(columns=['coverable','ipc_empalme_ipim']), 
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                    'año_mes': st.column_config.DateColumn("Año-Mes", format="YYYY-MM"),
                    'usd_blue': 'Valor USD blue',
                    'años_riesgos_total': st.column_config.NumberColumn("Años Riesgo", format="%.0f"),
                    'plan': 'Plan',
                    "suma_asegurada": st.column_config.NumberColumn("SA", format="$ %.0f"),
                    "suma_asegurada_ipc": st.column_config.NumberColumn("SA IPC", format="$ %.0f"),
                    "suma_asegurada_exp_a_riesgos": st.column_config.NumberColumn("SA Exp. a Riesgos", format="$ %.0f"),
                    "sa_prom": st.column_config.NumberColumn("SA Prom", format="$ %.0f"),
                    "var_sa_prom": st.column_config.NumberColumn(
                        "Var. SA Prom %", 
                        format="%.2f%%",
                    ),
                    "suma_asegurada_exp_a_riesgos_ipc": st.column_config.NumberColumn("SA Exp. a Riesgos IPC", format="$ %.0f"),
                    "sa_prom_ipc": st.column_config.NumberColumn("SA Prom IPC", format="$ %.0f"),
                    "var_sa_prom_ipc": st.column_config.NumberColumn(
                        "Var. SA Prom IPC %", 
                        format="%.2f%%",
                    ),
                    "suma_asegurada_usd": st.column_config.NumberColumn("SA USD", format="$ %.0f"),
                    "suma_asegurada_exp_a_riesgos_usd": st.column_config.NumberColumn("SA Exp. a Riesgos USD", format="$ %.0f"),
                    "sa_prom_usd": st.column_config.NumberColumn("SA Prom USD", format="$ %.0f"),
                    "var_sa_prom_usd": st.column_config.NumberColumn(
                        "Var. SA Prom USD %", 
                        format="%.2f%%",
                        ),
                    "costo_prom_ipc": st.column_config.NumberColumn("Coste medio rep. IPC", format="$ %.0f"),
                    "sa_prom_ipc": st.column_config.NumberColumn("SA prom. IPC", format="$ %.0f"),
                    "var_costo_prom_ipc": st.column_config.NumberColumn(                            
                        "Var. Costo rep. IPC %", 
                        format="%.2f%%",
                        ),
                    "costo_pieza_prom_hist": st.column_config.NumberColumn("CM rep. hist", format="$ %.0f"),
                    "costo_prom_ipc": st.column_config.NumberColumn("CM rep. IPC", format="$ %.0f"),
                    "costo_prom_usd": st.column_config.NumberColumn("CM rep. USD", format="$ %.0f"),
                    "var_costo_prom_usd": st.column_config.NumberColumn(
                        "Var. Costo rep. USD %", 
                        format="%.2f%%",
                        ),
                    "var_costo_prom_hist": st.column_config.NumberColumn(
                        "Var. Costo rep. %", 
                        format="%.2f%%",
                        ),

                    "var_ipc": st.column_config.NumberColumn(
                        "Var. IPC %", 
                        format="%.2f%%",
                        ),
                    })    
        with st.container(border=True):

            st.markdown("#### 3. Comparativo por marcas: Variación SA (Plan 30) vs Costo medio repuestos") 

            tab1, tab2, tab3 = st.tabs(["📊 Histórico", "📈 IPC", "💵 USD"])

            with tab1:
                fig_marcas_h = generar_grafico_comparativo_facetado(
                    df_sa_rep_cartera_marcas, 
                    'var_costo_prom_hist', 
                    'var_sa_prom', 
                    "Variación Mensual: Histórico",
                    col_ref='var_ipc' 
                )

                st.plotly_chart(fig_marcas_h, use_container_width=True)

            with tab2:
                fig_marcas_ipc = generar_grafico_comparativo_facetado(
                    df_sa_rep_cartera_marcas, 
                    'var_costo_prom_ipc', 
                    'var_sa_prom_ipc', 
                    "Variación Mensual: IPC",
                )
                st.plotly_chart(fig_marcas_ipc, use_container_width=True)

            with tab3:
                fig_marcas_usd = generar_grafico_comparativo_facetado(
                    df_sa_rep_cartera_marcas, 
                    'var_costo_prom_usd', 
                    'var_sa_prom_usd', 
                    "Variación Mensual: USD"
                )
                st.plotly_chart(fig_marcas_usd, use_container_width=True)


        st.markdown('')
        st.markdown("#### Data Cruda")
        st.dataframe(df_sa_rep_cartera_marcas.drop(columns=['coverable','plan','ipc_empalme_ipim']), 
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                    'año_mes': st.column_config.DateColumn("Año-Mes", format="YYYY-MM"),
                    'usd_blue': 'Valor USD blue',
                    'años_riesgos_total': st.column_config.NumberColumn("Años Riesgo", format="%.0f"),
                    "suma_asegurada": st.column_config.NumberColumn("SA", format="$ %.0f"),
                    "suma_asegurada_ipc": st.column_config.NumberColumn("SA IPC", format="$ %.0f"),
                    "suma_asegurada_exp_a_riesgos": st.column_config.NumberColumn("SA Exp. a Riesgos", format="$ %.0f"),
                    "sa_prom": st.column_config.NumberColumn("SA Prom", format="$ %.0f"),
                    "var_sa_prom": st.column_config.NumberColumn(
                        "Var. SA Prom %", 
                        format="%.2f%%",
                    ),
                    "suma_asegurada_exp_a_riesgos_ipc": st.column_config.NumberColumn("SA Exp. a Riesgos IPC", format="$ %.0f"),
                    "sa_prom_ipc": st.column_config.NumberColumn("SA Prom IPC", format="$ %.0f"),
                    "var_sa_prom_ipc": st.column_config.NumberColumn(
                        "Var. SA Prom IPC %", 
                        format="%.2f%%",
                    ),
                    "suma_asegurada_usd": st.column_config.NumberColumn("SA USD", format="$ %.0f"),
                    "suma_asegurada_exp_a_riesgos_usd": st.column_config.NumberColumn("SA Exp. a Riesgos USD", format="$ %.0f"),
                    "sa_prom_usd": st.column_config.NumberColumn("SA Prom USD", format="$ %.0f"),
                    "var_sa_prom_usd": st.column_config.NumberColumn(
                        "Var. SA Prom USD %", 
                        format="%.2f%%",
                        ),
                    "costo_prom_ipc": st.column_config.NumberColumn("Coste medio rep. IPC", format="$ %.0f"),
                    "sa_prom_ipc": st.column_config.NumberColumn("SA prom. IPC", format="$ %.0f"),
                    "var_costo_prom_ipc": st.column_config.NumberColumn(                            
                        "Var. Costo rep. IPC %", 
                        format="%.2f%%",
                        ),
                    "costo_pieza_prom_hist": st.column_config.NumberColumn("CM rep. hist", format="$ %.0f"),
                    "costo_prom_ipc": st.column_config.NumberColumn("CM rep. IPC", format="$ %.0f"),
                    "costo_prom_usd": st.column_config.NumberColumn("CM rep. USD", format="$ %.0f"),
                    "var_costo_prom_usd": st.column_config.NumberColumn(
                        "Var. Costo rep. USD %", 
                        format="%.2f%%",
                        ),
                    "var_costo_prom_hist": st.column_config.NumberColumn(
                        "Var. Costo rep. %", 
                        format="%.2f%%",
                        ),

                    "var_ipc": st.column_config.NumberColumn(
                        "Var. IPC %", 
                        format="%.2f%%",
                        ),
                    })    
        
    # ======================================================================================
    # --- VARIACION CM RUEDAS - GRANT ======================================================
    # ======================================================================================

    elif current_analysis == opcion_10:

        def create_plot_grant(df, y_col, y_label, titulo, x_tickangle=None):       
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
                        color_discrete_sequence=["#FB0D0D", "lightgreen", "blue", "gray", "magenta", "cyan", "orange", '#2CA02C'],
                        title=titulo,
                        labels={'año_mes': '', y_col: y_label,}# 'marca': 'Marca', 'tipo_cristal': 'Tipo de Cristal'}
                    )

                    fig.update_layout(
                        height=400, # Altura del subplot individual
                        font=dict(family="Arial", size=15),
                        margin=dict(t=50, b=0, l=0, r=0),
                    )
                    fig.for_each_xaxis(
                    lambda xaxis: xaxis.update(
                        tickangle=x_tickangle, # Aplicar el ángulo deseado
                        showticklabels=True     # Asegurarse de que las etiquetas sean visibles (si fuera necesario)
                        )
                    )
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                    
                    return fig
        

        def generar_grafico_gap_interanual(df, col_valor, label, titulo):
            """
            Genera un gráfico comparativo de Año Actual vs Año Anterior (Gap Analysis).
            El eje X son los meses (1-12) y las líneas son los años.
            """
            df_gap = df[df.año!=2026].copy()
            df_gap['año_mes'] = pd.to_datetime(df_gap['año_mes'])
            
            
            # Normalización a porcentaje si es necesario
            if df_gap[col_valor].abs().max() <= 1.1:
                df_gap[col_valor] = df_gap[col_valor] * 100

            # Ordenamos por mes para asegurar la continuidad de la línea
            df_gap = df_gap.sort_values('mes')

            # Diccionario de meses para el eje X
            meses_nombres = {
                1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
            }
            df_gap['Mes_Nom'] = df_gap['mes'].map(meses_nombres)

            fig = px.line(
                df_gap,
                x='mes',
                y=col_valor,
                color='año',
                text=col_valor,
                markers=True,
                title=titulo,
                labels={col_valor: label, 'mes': ''},
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            # Formato de hover y etiquetas
            fig.update_traces(
                hovertemplate='<b>Año-Mes:</b> %{x} %{fullData.name} <br><b>Coste prom:</b> $ %{y:,.0f}<extra></extra>',
                texttemplate='$ %{text:,.0f}', # Muestra 2 decimales. Usa %{text:,.0f} para miles sin decimales.
                textposition='top center', # Posiciona el valor arriba del punto
                textfont_size=12,
                cliponaxis=False # Evita que las etiquetas se corten en los bordes
                )

            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(meses_nombres.keys()),
                    ticktext=list(meses_nombres.values())
                ),
                # yaxis=dict(tickformat=".2f", ticksuffix="%"),
                hovermode="x unified",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=None),
                margin=dict(t=100, b=40, l=40, r=40)
            )

            return fig

        
        st.markdown("## Evolutivo: Coste medio de Ruedas")
        st.markdown("##### _Fuente: Grant_\n_Período: enero 2023 - febrero 2026_")
        st.markdown("---")


        st.markdown("### Costo promedio siniestro de Ruedas (con IVA)")

        fig_grant_hist = create_plot_grant(df_ruedas_grant, 'costo_medio', 'Costo Promedio ARS', 'Costo promedio siniestro de ruedas con IVA en AR$')
        fig_grant_ipc = create_plot_grant(df_ruedas_grant, 'costo_medio_ipc', 'Costo Promedio ajust. IPC', 'Costo promedio siniestro de ruedas con IVA ajust. IPC')
        fig_grant_usd = create_plot_grant(df_ruedas_grant, 'costo_medio_usd', 'Costo Promedio USD', 'Costo promedio siniestro de ruedas con IVA en USD')
        fig_grant_vs_ipc = create_plot_grant(df_ruedas_grant, 'var_cm_ars_%', 'Var. Costo Promedio ARS', 'Var. costo promedio ruedas vs var. IPC')

        fig_grant_vs_ipc.data[0].name = "Var. CM Ruedas"
        fig_grant_vs_ipc.data[0].showlegend = True
        fig_grant_vs_ipc.add_trace(go.Scatter(
            x=df_neum_grant['año_mes'],
            y=df_neum_grant['var_ipc'],
            name='IPC',        # Nombre que aparecerá en la leyenda
            mode='lines',
            line=dict(color='white', dash='dot')
        ))
        fig_grant_vs_ipc.update_traces(
            hovertemplate='<b>%{fullData.name}</b>: %{y:.2f}%<extra></extra>')
        fig_grant_vs_ipc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_grant_vs_ipc.update_layout(hovermode="x unified",
                                      yaxis=dict(tickformat=".2f", ticksuffix="%"))


        tab1, tab2, tab3, tab4 = st.tabs(["📊 Histórico", "📈 IPC", "💵 USD", "🔺Variación vs IPC"])
        with tab1:
            st.plotly_chart(fig_grant_hist, use_container_width=True)
        with tab2: 
            st.plotly_chart(fig_grant_ipc, use_container_width=True)
        with tab3: 
            st.plotly_chart(fig_grant_usd, use_container_width=True)
        with tab4: 
            st.plotly_chart(fig_grant_vs_ipc, use_container_width=True)


        tab1, tab2, tab3 = st.tabs(["📊 Histórico", "📈 IPC", "💵 USD"])
        with tab1:
            fig_gap = generar_grafico_gap_interanual(df_ruedas_grant, 'costo_medio','Coste Medio ARS', "Gap Interanual: Costo promedio siniestro de ruedas histórico")
            st.plotly_chart(fig_gap, use_container_width=True)
        with tab2:
            fig_gap_ipc = generar_grafico_gap_interanual(df_ruedas_grant, 'costo_medio_ipc','Coste Medio IPC', "Gap Interanual: Costo promedio siniestro de ruedas ajust. IPC")
            st.plotly_chart(fig_gap_ipc, use_container_width=True)
        with tab3:
            fig_gap_usd = generar_grafico_gap_interanual(df_ruedas_grant, 'costo_medio_usd','Coste Medio USD', "Gap Interanual: Costo promedio siniestro de ruedas en USD")
            st.plotly_chart(fig_gap_usd, use_container_width=True)

        with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
            st.dataframe(df_ruedas_grant.drop(columns=['mes','año']), 
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                        'año_mes': st.column_config.DateColumn("Año-Mes", format="YYYY-MM"),
                        'usd_blue': 'Valor USD blue',
                        "costo_medio": st.column_config.NumberColumn("Coste Medio ruedas", format="$ %.0f"),
                        "costo_medio_ipc": st.column_config.NumberColumn("Coste Medio IPC", format="$ %.0f"),
                        "costo_medio_usd": st.column_config.NumberColumn("Coste Medio USD", format="$ %.0f"),
                        "ipc": st.column_config.NumberColumn("IPC", format="%.2f"),
                        "var_cm_ars_%": st.column_config.NumberColumn(
                            "Var. CM histórico %", 
                            format="%.2f%%",
                        ),
                        "var_cm_ipc_%": st.column_config.NumberColumn(
                            "Var. CM IPC %", 
                            format="%.2f%%",
                        ),
                        "var_cm_usd_%": st.column_config.NumberColumn(
                            "Var. CM en USD %", 
                            format="%.2f%%",
                            ),
                        "var_ipc": st.column_config.NumberColumn(
                            "Var. IPC %", 
                            format="%.2f%%",
                            ),
                        })    

        st.subheader('', divider='grey')


        st.markdown("#### Variación del valor de un neumático promedio (Para todas las marcas y medidas de mayor rotación) - MERCADO ASEGURADOR")

        fig_neum_hist = create_plot_grant(df_neum_grant, 'costo_medio', 'Costo Promedio ARS', 'Variación costo promedio de un neumático en AR$')
        fig_neum_ipc = create_plot_grant(df_neum_grant, 'costo_medio_ipc', 'Costo Promedio ajust. IPC', 'Variación costo promedio de un neumático ajust. IPC')
        fig_neum_usd = create_plot_grant(df_neum_grant, 'costo_medio_usd', 'Costo Promedio USD', 'Variación costo promedio de un neumático en USD')
        fig_neum_vs_ipc = create_plot_grant(df_neum_grant, 'var_%', 'Var. Costo Promedio ARS', 'Var. costo promedio neumático vs var. IPC')

        fig_neum_vs_ipc.data[0].name = "Var. CM Neumático"
        fig_neum_vs_ipc.data[0].showlegend = True
        fig_neum_vs_ipc.add_trace(go.Scatter(
            x=df_neum_grant['año_mes'],
            y=df_neum_grant['var_ipc'],
            name='IPC',        # Nombre que aparecerá en la leyenda
            mode='lines',
            line=dict(color='white', dash='dot')
        ))
        fig_neum_vs_ipc.update_traces(
            hovertemplate='<b>%{fullData.name}</b>: %{y:.2f}%<extra></extra>')
        fig_neum_vs_ipc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_neum_vs_ipc.update_layout(hovermode="x unified",
                                      yaxis=dict(tickformat=".2f", ticksuffix="%"))
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Histórico", "📈 IPC", "💵 USD", "🔺Variación vs IPC"])
        with tab1:
            st.plotly_chart(fig_neum_hist, use_container_width=True)
        with tab2: 
            st.plotly_chart(fig_neum_ipc, use_container_width=True)
        with tab3:     
            st.plotly_chart(fig_neum_usd, use_container_width=True)
        with tab4:
            st.plotly_chart(fig_neum_vs_ipc, use_container_width=True)
        


        tab1, tab2, tab3 = st.tabs(["📊 Histórico", "📈 IPC", "💵 USD"])
        with tab1:
            fig_neum_gap = generar_grafico_gap_interanual(df_neum_grant, 'costo_medio','Coste Medio ARS', "Gap Interanual: Costo promedio neumático histórico")
            st.plotly_chart(fig_neum_gap, use_container_width=True)
        with tab2:
            fig_neum_gap_ipc = generar_grafico_gap_interanual(df_neum_grant, 'costo_medio_ipc','Coste Medio IPC', "Gap Interanual: Costo promedio neumático ajust. IPC")
            st.plotly_chart(fig_neum_gap_ipc, use_container_width=True)
        with tab3:
            fig_neum_gap_usd = generar_grafico_gap_interanual(df_neum_grant, 'costo_medio_usd','Coste Medio USD', "Gap Interanual: Costo promedio neumático en USD")
            st.plotly_chart(fig_neum_gap_usd, use_container_width=True)

        with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
            st.dataframe(df_neum_grant.drop(columns=['mes','año']), 
                use_container_width=True,
                hide_index=True,
                column_config={
                'año_mes': st.column_config.DateColumn("Año-Mes", format="YYYY-MM"),
                'usd_blue': 'Valor USD blue',
                "costo_medio": st.column_config.NumberColumn("Coste Medio neumáticos", format="$ %.0f"),
                "costo_medio_ipc": st.column_config.NumberColumn("Coste Medio IPC", format="$ %.0f"),
                "costo_medio_usd": st.column_config.NumberColumn("Coste Medio USD", format="$ %.0f"),
                "ipc": st.column_config.NumberColumn("IPC", format="%.2f"),
                "var_%": st.column_config.NumberColumn(
                    "Var. CM histórico %", 
                    format="%.2f%%",
                ),
                "var_cm_ipc_%": st.column_config.NumberColumn(
                    "Var. CM IPC %", 
                    format="%.2f%%",
                ),
                "var_cm_usd_%": st.column_config.NumberColumn(
                    "Var. CM en USD %", 
                    format="%.2f%%",
                    ),
                "var_ipc": st.column_config.NumberColumn(
                    "Var. IPC %", 
                    format="%.2f%%",
                    ),
                })    
            
    # ======================================================================================
    # --- INDICE - CONSTRUCCION ============================================================
    # ======================================================================================
            
    elif current_analysis == opcion_11:
        st.title('Ponderacioens y cálculo del Indice')
        st.markdown("#### _En construcción..._ ")
        # st.markdown("#### _Fuente de datos: Orion/Cesvi_ \nFecha actualización: **diciembre 2025**")

    #     def plot_indices_comparados(df, title):
    #         # Lista de las variables que quieres graficar
    #         columnas_indices = [
    #             'indice_cascos', 
    #             'indice_daño_mat', 
    #             'indice_cristales', 
    #             'indice_ruedas', 
    #             'indice_autos'
    #         ]
            
    #         # Asegúrate de que el dataframe esté ordenado por fecha
    #         df = df.sort_values('año_mes')

    #         # Crear el gráfico (Formato "Wide")
    #         fig = px.line(
    #             df, 
    #             x='año_mes', 
    #             y=columnas_indices,
    #             labels={'año_mes': 'Período', 'value': 'Índice', 'variable': 'Indicador'},
    #             title=title,
    #             markers=True # Puntos en cada mes para mayor precisión
    #         )

    #         # Configuración estética y Hover Unificado
    #         fig.update_layout(
    #             height=600,
    #             hovermode="x unified", # Ver todos los índices al mismo tiempo
    #             # plot_bgcolor="white",
    #             legend=dict(
    #                 orientation="h",
    #                 yanchor="top",
    #                 y=-0.2,
    #                 xanchor="center",
    #                 x=0.5,
    #                 title_text=""
    #             ),
    #             margin=dict(t=80, b=100, l=50, r=20),
    #             title=dict(font=dict(size=22, family="Arial Black"), x=0.5)
    #         )

    #         # Mejorar el hover para que muestre 2 decimales
    #         fig.update_traces(
    #             mode="lines+markers",
    #             marker=dict(size=4),
    #             hovertemplate="<b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>"
    #         )

    #         # Cuadrícula sutil
    #         # fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", tickangle=45)
    #         # fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    #         return fig
        
    #     with st.expander("Ver tabla de datos históricos", icon=":material/query_stats:"):
    #         st.dataframe(indice_hist, hide_index=True, use_container_width=True,)

    #     fig_indice_hist = plot_indices_comparados(indice_hist, 'Indice Histórico')
    #     st.plotly_chart(fig_indice_hist, use_container_width=True)

    #     with st.expander("Ver tabla de datos ajust. IPC", icon=":material/query_stats:"):
    #         st.dataframe(indice_ipc, hide_index=True, use_container_width=True,)
    #     fig_indice_ipc = plot_indices_comparados(indice_ipc, 'Indice ajust. por IPC')
    #     st.plotly_chart(fig_indice_ipc, use_container_width=True)

    #     with st.expander("Ver tabla de datos en USD", icon=":material/query_stats:"):
    #         st.dataframe(indice_usd, hide_index=True, use_container_width=True,)
    #     fig_indice_usd = plot_indices_comparados(indice_usd, 'Indice en USD')
    #     st.plotly_chart(fig_indice_usd, use_container_width=True)








        



    
