import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # Necesario para go.Figure en caso de datos vacíos
from plotly.subplots import make_subplots
import json
import requests
from io import StringIO
import unicodedata
import locale

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
    df_cristal = pd.read_csv('data/base_pkt_ok.csv')
    df_pagos_cristal = pd.read_csv('data/pagos_pkt_ok.csv')

    # dfs de repuestos orion/cesvi
    df_tipo_rep = pd.read_csv('data/df_tipo_rep_oct.csv')
    df_rep_tv = pd.read_csv('data/df_rep_tv_oct.csv')

    # dfs de mano de obra orion/cesvi
    df_cm_mo = pd.read_csv('data/df_cm_mo_oct.csv')

    # dfs mano de obra cleas si/cleas no
    df_cm_mo_cleas = pd.read_csv('data/df_cm_mo_cleas_oct.csv')

    # dfs marcas
    df_marcas_autos = pd.read_csv('data/evol_todas_marcas_autos.csv')
    df_rtos_marca_mes = pd.read_csv('data/df_rtos_marca_mes_oct.csv')
    df_marcas_camiones = pd.read_csv('data/camion_marcas.csv')
    df_rtos_marca_mes_cam = pd.read_csv('data/df_rtos_marca_mes_cam_oct.csv')
    df_distrib_marcas_cartera = pd.read_csv('data/distrib_ar_marca_cartera.csv')

    # dfs var x prov
    df_cm_prov_orion = pd.read_csv('data/base_cm_prov_orion.csv')
    df_cm_prov = pd.read_csv('data/base_cm_prov_ok.csv')
    with open('data/prov.geojson', 'r', encoding='utf-8') as f:
        provincias_geojson = json.load(f)
    comparativo_orion_prov = pd.read_csv('data/comparativo_orion_prov.csv')
    comparativo_cm_siniestral = pd.read_csv('data/comparativo_cm_siniestral.csv')

    # dfs comparativo mano de obra
    df_chapa_pintura = pd.read_csv('data/df_chapa_pintura.csv')
    df_mo_repuestos_final = pd.read_csv('data/df_mo_repuestos_final_ok.csv')
    df_peritaciones = pd.read_csv('data/df_peritaciones.csv')
    df_costo_hora = pd.read_csv('data/df_costo_hora_ok.csv')
    df_tot_reparacion = pd.read_csv('data/df_tot_reparacion_ok.csv')

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

# ---- Formateo base provincias --------------------------------------------------
df_cm_agg = df_cm_prov.groupby(['coverable','año','provincia',]).agg(
    coste_medio_prom=('coste_medio', 'mean'))
df_cm_agg = df_cm_agg.reset_index()
# cambio formato de coste medio a int
df_cm_agg['coste_medio_prom'] = df_cm_agg['coste_medio_prom'].astype(int)

# ==========================================================================
df_pagos_cristal['tipo_cristal'] = df_pagos_cristal['tipo_cristal'].replace('Cristales lateral y techo', 'Cristales laterales y de techo')

# ---- Variables de sesión para mostrar/ocultar gráficos --------------------------------------------------
if 'show_pie_chart' not in st.session_state:
    st.session_state.show_pie_chart = False

if 'show_pie_chart_2' not in st.session_state:
    st.session_state.show_pie_chart_2 = False

if 'show_pie_chart_3' not in st.session_state:
    st.session_state.show_pie_chart_3 = False

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

opcion_1 = "Evolutivo lista de precios Pilkington"
opcion_2 = "Comparación precios vs pagos Cristales"
opcion_3 = "Evolutivo precios ORION/CESVI"
opcion_4 = "Análisis Coste Medio por Provincia"
opcion_5 = "Comparativo de Mano de Obra (L2 vs Cesvi vs Colegas)"
opcion_6 = "Evolutivo pagos Robo de Ruedas"

OPTIONS = [opcion_1, opcion_2, opcion_3, opcion_4, opcion_5, opcion_6]

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

    # Renderiza la pantalla del análisis basada en la selección actual
    # if current_analysis == opcion_1:
    #     st.title("Variación de Precios de Cristales y Mano de obra por Marca y Zona")
    #     st.markdown("#### _Fuente de datos: Listas de precios de Pilkington_")
    #     st.markdown("---")
        
    #     # Aquí continúa el código de tu análisis 1 (dropdowns, gráficos, etc.)
    #     st.success("Aquí iría el análisis completo de Pilkington...")
        
    # elif current_analysis == opcion_2:
    #     st.title("Evolutivo precios ORION/CESVI")
    #     st.info("Aquí iría el contenido del Análisis 2...")
        
    # # ... y así sucesivamente para opcion_3, opcion_4, opcion_5
    
    # # Mantenemos el divisor al final del contenido del análisis
    # st.markdown("---")


# selected_analysis = st.selectbox(
#     'Seleccionar Análisis:',
#     options=OPTIONS,
#     index=0,
#     label_visibility ='collapsed'
# )
# st.markdown("---")


# ==========================================================================
# ---- Análisis PILKINGTON -------------------------------------------------
# ==========================================================================
    if current_analysis == opcion_1:
        st.markdown("## Variación de precios de Cristales y Mano de obra por Marca y Zona")
        st.markdown("#### _Fuente de datos: Listas de precios de Pilkington_")
                    # \n:white_small_square: _La Segunda BI (pagos)_")
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

            # Ajustes del gráfico
            fig.update_layout(
                height=600, # Altura del subplot individual
                legend_title_text='Marca',
                font=dict(family="Arial", size=15),
                margin=dict(t=100, b=0, l=0, r=0),
                title=dict(
                font=dict(
                    size=24,  # <-- Aumenta este valor para un título más grande (ej: 24, 28, etc.)
                    family="Arial",
                    # color="black" # Opcional: puedes cambiar el color también
                ),
                # x=0.5, # Opcional: Centrar el título (0 es izquierda, 1 es derecha)
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

        # ----- GRAFICOS HISTORICOS, IPC y USD --------------------------------------------------

        if not selected_marcas:
            st.warning(":warning: Seleccionar una marca para ver la información.")
            st.stop()

        else:
            # muestro grafico torta MARCAS AUTOS 
            if st.button("Mostrar/Ocultar Distribución de Marcas Autos", icon='📊'):
                st.session_state.show_pie_chart_3 = not st.session_state.show_pie_chart_3
            
            if st.session_state.show_pie_chart_3:
                st.subheader('Distribución de Años Riesgo por Marca')
                fig_pie = create_pie_chart(df_distrib_marcas_cartera, 'años_riesgos_total')
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown('Total AR: ' + str(df_distrib_marcas_cartera['años_riesgos_total'].sum()))
                st.markdown('Total marcas: 49' )
                st.markdown("---")

                st.markdown('')
    # ==========================================================================

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
        
                fig1_ipc_.update_layout(legend_title_text='Variación')

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
# ---- Comparación PAGOS vs LISTA PRECIOS ----------------------------------
# ==========================================================================

    elif current_analysis == opcion_2:
        df_cristal_copy = df_cristal.copy()
        df_cristal_copy.rename(columns={
            'marca': 'marca_vehiculo',
        }, inplace=True)

        df_cristal_copy['precio_total'] = df_cristal_copy['precio'] + df_cristal_copy['instalacion'] 
        df_cristal_copy['precio_total_ipc'] = df_cristal_copy['precio_ipc'] + df_cristal_copy['instalacion_ipc'] 
        df_cristal_copy['precio_total_usd'] = df_cristal_copy['precio_usd'] + df_cristal_copy['instalacion_usd']
        # df_cristal_copy = df_cristal_copy[(df_cristal_copy['zona'] == 'ROSARIO')]

        # df_cristal_copy_final = df_cristal_copy.groupby(['fecha', 'tipo_cristal', 'marca_vehiculo','zona']).agg(
        #     {'precio': 'mean',
        #     'instalacion': 'mean',
        #     'precio_ipc': 'mean',
        #     'instalacion_ipc': 'mean',
        #     'precio_usd': 'mean',
        #     'instalacion_usd': 'mean'}
        # ).reset_index()


        def create_plot_pagos(df_source, y1, y2, y3, title, x_tickangle=45):

            df_filtered = df_source[
                (df_source['tipo_cristal'] == selected_cristal) 
                # & (df_source['marca_vehiculo'].isin(selected_marcas)) 
                # & (df_source['zona'].isin(selected_zone))
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


            fig.update_yaxes(title_text=f"{y1_label}", secondary_y=False, nticks=10, showgrid=True)
            fig.update_yaxes(title_text=f"{y2_label}", secondary_y=True, nticks=10, showgrid=False)

            # Ajustes del Gráfico
            fig.update_layout(
                title_text=title,
                height=700,
                legend_title_text='', # Dejar vacío ya que el nombre de la línea lo explica
                font=dict(family="Arial", size=15),
                margin=dict(t=100, b=0, l=0, r=0),
                title=dict(
                    font=dict(size=20, family="Arial"),
                    # x=0.5, # Centrar título
                    # xanchor='center'
                ),
                # legend=dict(
                #     orientation="h",
                #     yanchor="top",
                #     y=-0.1, # Mover leyenda debajo del gráfico
                #     xanchor="center",
                #     x=0.5
                # )
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
                (df['tipo_cristal'] == selected_cristal) &
                (df['marca_vehiculo'].isin(selected_marcas)) 
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
                # line_group='marca',
                #title='', agrego titulo con subheader
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
        
        def create_plot_precios(df, y_col, y_label, title, fixed_color, x_tickangle=None):
            if df.empty:
                fig = go.Figure().update_layout(
                    title_text=f"No hay datos para graficar",
                    height=400,
                    font=dict(family="Arial", size=10),
                    title_font_size=12
                )
                return fig

            df_filtered = df[
                (df['tipo_cristal'] == selected_cristal) &
                (df['marca_vehiculo'].isin(selected_marcas)) 
            ].copy()

            # --- Agregación ---
            # df_filtered.sort_values('fecha', inplace=True)
            df_plot = df_filtered.groupby(['fecha']).agg(
                {
                'precio_total': 'mean',
                'precio_total_ipc': 'mean',         
                'precio_total_usd': 'mean'        
                }).reset_index()

            fig = px.bar(
                df_plot,
                x='fecha',
                y=y_col,
                # color=color,
                # facet_col=facet_col,
                # Si tienes múltiples marcas por mes, considera usar barmode='group' o 'stack' aquí si es necesario
                # barmode='group', 
                labels={'fecha': '', y_col: y_label,}
            )
            fig.update_traces(marker_color=fixed_color)

            # --- Ajustes del Layout (Se mantienen) ---
            fig.update_layout(
                title_text=title,
                height=500, 
                width=900, 
                legend_title_text='',
                font=dict(family="Arial", size=15),
                margin=dict(t=50, b=0, l=0, r=0),
                title=dict(
                    font=dict(size=20, family="Arial") 
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top", 
                    y=-0.2,
                    xanchor="center",
                    x=0.5)
            )

            # --- Ajustes de Eje X y Facetas (Se mantienen) ---
            fig.for_each_xaxis(
                lambda xaxis: xaxis.update(
                    tickangle=x_tickangle, 
                    showticklabels=True
                    )
            )
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            
            # # Calcula el rango real de tus datos
            # y_min = df_plot[y_col].min() * 0.95  # 5% debajo del mínimo
            # y_max = df_plot[y_col].max() * 1.05  # 5% por encima del máximo

            # fig.update_yaxes(
            #     range=[y_min, y_max],  # Define el rango manualmente
            #     # Si tienes subplots (facet_col), usa: row='all', col='all'
            #     nticks=6,
            # )
            # # fig.update_yaxes(title_text=f"{y1_label}", secondary_y=False, nticks=10, showgrid=True)

            return fig
        
        available_cristales = sorted(df_pagos_cristal['tipo_cristal'].unique().tolist())       
        available_marcas = sorted(df_pagos_cristal['marca_vehiculo'].unique().tolist())
        DEFAULT_MARCAS = ["VOLKSWAGEN", "CHEVROLET", "FORD",  "TOYOTA", "FIAT", "PEUGEOT", "RENAULT"]

        st.markdown("## Comparación pagos L2 vs lista precios de Cristales")
        st.markdown("#### _Fuente de datos:_ \
                    \n:white_small_square: _Listas de precios de Pilkington_ \
                    \n:white_small_square: _La Segunda BI (pagos)_")
        
        st.markdown("---")
        with st.sidebar:
            st.markdown("---")
            st.markdown("Filtros") # Título para la barra lateral
            st.markdown("##### _Seleccionar Tipo de Cristal:_") 
            selected_cristal = st.selectbox(
                "",
                options=available_cristales,
                index=1,
                label_visibility ='collapsed',
                # placeholder= "Selecciona una opción...",
            )
            st.markdown("---")

            # st.markdown("##### _Seleccionar Marcas:_")
            # selected_marcas_raw = st.multiselect(
            #     "Marcas",
            #     options=available_marcas_con_todas,
            #     default=[m for m in DEFAULT_MARCAS if m in available_marcas],
            #     # default=["TODAS LAS MARCAS"],
            #     label_visibility='collapsed',
            # )
            # st.markdown("---")

        #     st.markdown("##### _Seleccionar Zona:_")
        #     selected_zone_raw = st.multiselect(
        #         "Zona",
        #         options=available_zonas_con_todas,
        #         default = "TODAS LAS ZONAS",
        #         label_visibility ='collapsed',

        #     )
        #     st.markdown("---")

        # if "TODAS LAS MARCAS" in selected_marcas_raw:
        #     selected_marcas = available_marcas 
        # else:
        #     selected_marcas = selected_marcas_raw
        # if "TODAS LAS ZONAS" in selected_zone_raw:
        #     selected_zone = available_zones
        # else:
        #     selected_zone = selected_zone_raw

        # if not selected_marcas:
        #     st.warning(":warning: Seleccionar al menos una Marca para ver la información.")
        #     st.stop()
        # if not selected_zone:
        #     st.warning(":warning: Seleccionar una opción de Zona para ver la información.")
        #     st.stop()

# ==== GRAFICO 1 - PAGOS PROMEDIO POR TIPO DE CRISTAL ===========================================
        fig_pagos1 = create_plot_pagos(
            df_pagos_cristal, 
            'monto_transaccion',
            'pago_ipc',
            'pago_usd',
            title=f'Pagos promedio (L2) - {selected_cristal}', 
            x_tickangle=45
        )
        st.plotly_chart(fig_pagos1, use_container_width=True)

        # tabla para mostrar:
        df_resumen1 = df_pagos_cristal.sort_values('año_mes_fecha_pago')
        df_resumen1 = df_resumen1[(df_resumen1['tipo_cristal'] == selected_cristal)
                # & (df_resumen1['marca_vehiculo'].isin(selected_marcas))
            ]
        df_tabla = df_resumen1.groupby(['año_mes_fecha_pago']).agg(
            {'monto_transaccion': 'mean',
            'pago_usd': 'mean',           
            'pago_ipc': 'mean'}).reset_index()
        df_tabla.monto_transaccion = df_tabla.monto_transaccion.round(0).astype(int) 
        df_tabla.pago_usd = df_tabla.pago_usd.round(0).astype(int) 
        df_tabla.pago_ipc = df_tabla.pago_ipc.round(0).astype(int)

        with st.expander("Ver tabla de datos (resumen)", icon=":material/query_stats:"):
            st.dataframe(df_tabla, hide_index=True, width=900)
        st.subheader('', divider='grey')
        

# ==== GRAFICO 2 - PAGOS PROM POR MARCA ===========================================
        col_1, col_2 = st.columns([0.9, 4])
        with col_1:
            st.markdown("**_Seleccionar Marcas:_**")
            selected_marcas = st.multiselect(
                "Marcas",
                options=available_marcas,
                default=[m for m in DEFAULT_MARCAS if m in available_marcas],
                # default=["TODAS LAS MARCAS"],
                label_visibility='collapsed',
            )

        with col_2:
            fig_pagos2_hist = create_plot_pagos_marcas(
                df_pagos_cristal,
                'monto_transaccion', 
                'marca_vehiculo',
                None,
                'Monto histórico',
                title=f'Pagos históricos por marca - {selected_cristal}', 
                x_tickangle=45)
            fig_pagos2_ipc = create_plot_pagos_marcas(
                df_pagos_cristal,
                'pago_ipc', 
                'marca_vehiculo',
                None,
                'Monto IPC',
                title=f'Pagos por marca ajustados por IPC - {selected_cristal}', 
                x_tickangle=45)
            fig_pagos2_usd = create_plot_pagos_marcas(
                df_pagos_cristal,
                'pago_usd', 
                'marca_vehiculo',
                None,
                'Monto USD',
                title=f'Pagos por marca en valor USD - {selected_cristal}', 
                x_tickangle=45)


            tab1, tab2, tab3 = st.tabs(["Histórico", "IPC", 'USD'])
            with tab1:
                st.plotly_chart(fig_pagos2_hist, use_container_width=True)
            with tab2:
                st.plotly_chart(fig_pagos2_ipc, use_container_width=True)
            with tab3:
                st.plotly_chart(fig_pagos2_usd, use_container_width=True)
                    
        df_resumen2 = df_pagos_cristal.sort_values(by=['año_mes_fecha_pago','marca_vehiculo'])
        df_resumen2 = df_resumen2[(df_resumen2['tipo_cristal'] == selected_cristal)
                & (df_resumen2['marca_vehiculo'].isin(selected_marcas))
            ]
        df_tabla2 = df_resumen2.groupby(['año_mes_fecha_pago','marca_vehiculo']).agg(
            {'monto_transaccion': 'mean',
            'pago_usd': 'mean',           
            'pago_ipc': 'mean'}).reset_index()
        
        df_tabla2.monto_transaccion = df_tabla2.monto_transaccion.round(0).astype(int) 
        df_tabla2.pago_usd = df_tabla2.pago_usd.round(0).astype(int) 
        df_tabla2.pago_ipc = df_tabla2.pago_ipc.round(0).astype(int)

        with st.expander("Ver tabla de datos (resumen)", icon=":material/query_stats:"):
            st.dataframe(df_tabla2, hide_index=True, width=900)

        st.subheader('', divider='grey')

# ==== GRAFICO 3 - PRECIOS DE LISTA PROM ===========================================

        COLOR_HIST = '#1f77b4'  # Azul
        COLOR_IPC = '#ff7f0e'   # Naranja
        COLOR_USD = '#2ca02c'   # Verde
                
        fig_precio_hist = create_plot_precios(
            df_cristal_copy, 
            'precio_total',
            'Precio (ARS)',
            title=f'Precios de lista promedio histórico - {selected_cristal}',
            fixed_color=COLOR_HIST, 
            x_tickangle=45
        )
        fig_precio_ipc = create_plot_precios(
            df_cristal_copy, 
            'precio_total_ipc',
            'Precio IPC (ARS)',
            title=f'Precios de lista promedio ajustado IPC - {selected_cristal}', 
            fixed_color=COLOR_IPC,
            x_tickangle=45
        )
        fig_precio_usd = create_plot_precios(
            df_cristal_copy, 
            'precio_total_usd',
            'Precio (USD)',
            title=f'Precios de lista promedio en valor USD - {selected_cristal}', 
            fixed_color=COLOR_USD,
            x_tickangle=45
        )

        tab1, tab2, tab3 = st.tabs(["Histórico", "IPC", 'USD'])
        with tab1:
            st.plotly_chart(fig_precio_hist, use_container_width=False)
        with tab2:
            st.plotly_chart(fig_precio_ipc, use_container_width=False)
        with tab3:
            st.plotly_chart(fig_precio_usd, use_container_width=False)
        st.write(f'Marcas seleccionadas: {", ".join(selected_marcas)}')

        # tabla para mostrar:
        df_resumen3 = df_cristal_copy.sort_values('fecha')
        df_resumen3 = df_resumen3[(df_resumen3['tipo_cristal'] == selected_cristal)
                & (df_resumen3['marca_vehiculo'].isin(selected_marcas))
            ]
        # df_resumen3.precio_total = df_resumen3.precio_total.round(0).astype(int) 
        # df_resumen3.precio_total_ipc = df_resumen3.precio_total_ipc.round(0).astype(int) 
        # df_resumen3.precio_total_usd = df_resumen3.precio_total_usd.round(0).astype(int)
        df_resumen3['fecha'] = df_resumen3['fecha'].dt.strftime('%Y-%m-%d')
        df_tabla3 = df_resumen3.groupby(['fecha']).agg(
            {
            'tipo_cristal': 'first',
            'precio': 'mean',
            'precio_ipc': 'mean',
            'precio_usd': 'mean',
            'instalacion': 'mean',
            'instalacion_ipc': 'mean',
            'instalacion_usd': 'mean',
            'precio_total': 'mean',
            'precio_total_ipc': 'mean',           
            'precio_total_usd': 'mean'}).reset_index()
        
        df_tabla3.precio_total = df_tabla3.precio_total.round(0).astype(int) 
        df_tabla3.precio_total_ipc = df_tabla3.precio_total_ipc.round(0).astype(int) 
        df_tabla3.precio_total_usd = df_tabla3.precio_total_usd.round(0).astype(int)
        df_tabla3.precio = df_tabla3.precio.round(0).astype(int) 
        df_tabla3.precio_ipc = df_tabla3.precio_ipc.round(0).astype(int) 
        df_tabla3.precio_usd = df_tabla3.precio_usd.round(0).astype(int)
        df_tabla3.instalacion = df_tabla3.instalacion.round(0).astype(int) 
        df_tabla3.instalacion_ipc = df_tabla3.instalacion_ipc.round(0).astype(int) 
        df_tabla3.instalacion_usd = df_tabla3.instalacion_usd.round(0).astype(int)

        with st.expander("Ver tabla de datos (resumen)", icon=":material/query_stats:"):
            st.dataframe(df_tabla3, hide_index=True)
        
        st.subheader('', divider='grey')


# === COMPARACION PRECIOS VS PAGOS ==========================================

        st.markdown(f"### **Comparación pagos vs precios de lista promedio - {selected_cristal}**")
        st.write('')

        df_comp_precios = df_cristal_copy.sort_values('fecha')
        df_comp_precios['fecha'] = pd.to_datetime(df_comp_precios['fecha'])
        df_comp_precios['fecha'] = df_comp_precios['fecha'].dt.strftime('%Y-%m')

        df_comp_pagos = df_pagos_cristal.sort_values(by=['año_mes_fecha_pago','marca_vehiculo'])
        df_comp_pagos['año_mes_fecha_pago'] = pd.to_datetime(df_comp_pagos['año_mes_fecha_pago'])
        df_comp_pagos['año_mes_fecha_pago'] = df_comp_pagos['año_mes_fecha_pago'].dt.strftime('%Y-%m')


        fechas_disponibles = sorted(df_comp_precios['fecha'].unique().tolist(), reverse=True)
        available_zones_full = sorted(df_comp_precios['zona'].unique().tolist())
        zona_a_excluir = 'ROSARIO' # O 'Rosario', usa el valor exacto que aparece en tu columna 'zona'
        available_zones_ui = [zona for zona in available_zones_full if zona != zona_a_excluir]

        available_zonas_con_todas = ["TODAS (general)"] + available_zones_ui

        col_fecha, col_marca, col_zona = st.columns(3)
        with col_fecha:
            selected_fecha = st.selectbox(
                ":arrow_right: Seleccione Fecha (AAAA-MM):",
                options=fechas_disponibles,
                index=0 # Por defecto, la fecha más reciente (debido al sorted(reverse=True))
            )

        with col_marca:
            selected_marcas_comp = st.selectbox(
                ":arrow_right: Seleccione Marca:",
                options=available_marcas,
                index=available_marcas.index("CHEVROLET") if "CHEVROLET" in available_marcas else 0,
                placeholder="Seleccione una Marca..."
            )

        with col_zona:
            selected_zone_raw = st.selectbox(
                ":arrow_right: Seleccione Provincia:",
                options=available_zonas_con_todas,
                index = available_zonas_con_todas.index("TODAS (general)"),
            )
        
        if "TODAS (general)" in selected_zone_raw:
            selected_zone = available_zones_full
        else:
            selected_zone = [selected_zone_raw]
        
        df_comp_precios = df_cristal_copy.sort_values('fecha')
        df_comp_precios['fecha'] = pd.to_datetime(df_comp_precios['fecha'])
        df_comp_precios['fecha'] = df_comp_precios['fecha'].dt.strftime('%Y-%m')

        df_comp_precios = df_comp_precios[(df_comp_precios['tipo_cristal'] == selected_cristal)
                & (df_comp_precios['marca_vehiculo'] == selected_marcas_comp)
                & (df_comp_precios['fecha'] == selected_fecha)
                & (df_comp_precios['zona'].isin(selected_zone))
            ]

        df_tabla3 = df_comp_precios.groupby(['fecha']).agg(
            {
            'tipo_cristal': 'first',
            'precio': 'mean',
            'precio_ipc': 'mean',
            'precio_usd': 'mean',
            'instalacion': 'mean',
            'instalacion_ipc': 'mean',
            'instalacion_usd': 'mean',
            'precio_total': 'mean',
            'precio_total_ipc': 'mean',           
            'precio_total_usd': 'mean'}).reset_index()
        
        df_tabla3.precio_total = df_tabla3.precio_total.round(0).astype(int) 
        df_tabla3.precio_total_ipc = df_tabla3.precio_total_ipc.round(0).astype(int) 
        df_tabla3.precio_total_usd = df_tabla3.precio_total_usd.round(0).astype(int)
        df_tabla3.precio = df_tabla3.precio.round(0).astype(int) 
        df_tabla3.precio_ipc = df_tabla3.precio_ipc.round(0).astype(int) 
        df_tabla3.precio_usd = df_tabla3.precio_usd.round(0).astype(int)
        df_tabla3.instalacion = df_tabla3.instalacion.round(0).astype(int) 
        df_tabla3.instalacion_ipc = df_tabla3.instalacion_ipc.round(0).astype(int) 
        df_tabla3.instalacion_usd = df_tabla3.instalacion_usd.round(0).astype(int)


        df_comp_pagos = df_comp_pagos[(df_comp_pagos['tipo_cristal'] == selected_cristal)
                & (df_comp_pagos['marca_vehiculo'] == selected_marcas_comp)
                & (df_comp_pagos['año_mes_fecha_pago'] == selected_fecha)
                & (df_comp_pagos['zona'].isin(selected_zone))
                ]
        
        df_tabla2 = df_comp_pagos.groupby(['año_mes_fecha_pago']).agg(
            {'monto_transaccion': 'mean',
            'pago_usd': 'mean',           
            'pago_ipc': 'mean'}).reset_index()
        
        df_tabla2.monto_transaccion = df_tabla2.monto_transaccion.round(0).astype(int) 
        df_tabla2.pago_usd = df_tabla2.pago_usd.round(0).astype(int) 
        df_tabla2.pago_ipc = df_tabla2.pago_ipc.round(0).astype(int)

        st.write('')

        def format_ars_value(number):
            """Formatea el número con punto como separador de miles y sin decimales."""
            # Retorna NaN o cadena vacía si el input no es numérico, para evitar errores
            if pd.isna(number):
                return ""
            
            # Convertir a entero (si es float) y usar el formato de coma (,) para miles
            formatted_number = f"{int(round(number)):,}"
            
            # Reemplazar la coma (,) por el punto (.) para ajustarse al estándar argentino
            return formatted_number.replace(',', '.')
        def format_ars_delta(diff_number):
            """Formatea la diferencia (delta) incluyendo signo (+/-) y punto de miles."""
            if pd.isna(diff_number):
                return "N/A"
                
            signo = '+' if diff_number > 0 else ('-' if diff_number < 0 else '')
            
            # Obtener el valor absoluto
            abs_number = abs(diff_number)
            
            # Usar el formateador de miles (punto) creado anteriormente
            valor_absoluto_formateado = format_ars_value(abs_number)
            
            # Construir la cadena final
            return f"{signo} ${valor_absoluto_formateado}"
        
        if  not df_tabla2.empty:
            # Agrupamos por los filtros para obtener el promedio de la métrica
            pago_promedio_ars = df_tabla2['monto_transaccion'].values[0]
            pago_promedio_ipc = df_tabla2['pago_ipc'].values[0]
            pago_promedio_usd = df_tabla2['pago_usd'].values[0]
        
            # Creamos un indicador en Streamlit para el precio
            st.markdown(f"#### 💰 Pago promedio (L2)")
            col_pago_ars, col_pago_ipc, col_pago_usd = st.columns(3)

            valor1 = format_ars_value(pago_promedio_ars)
            valor2 = format_ars_value(pago_promedio_ipc)
            valor3 = format_ars_value(pago_promedio_usd)

            with col_pago_ars:
                st.metric(label="Pago (ARS)", value=f'${valor1}', border=True)
            with col_pago_ipc:
                st.metric(label="Pago IPC (ARS)", value=f"${valor2}",border=True)
            with col_pago_usd:
                st.metric(label="Pago (USD)", value=f"U$D {valor3}",border=True)
            
        else:
            st.warning(":warning: No se encontraron datos de **Pagos** para la selección actual.")
            pago_promedio_ars = None

        if  not df_tabla3.empty:
            precio_promedio_ars = df_tabla3['precio_total'].values[0]
            precio_promedio_ipc = df_tabla3['precio_total_ipc'].values[0]
            precio_promedio_usd = df_tabla3['precio_total_usd'].values[0]
        
            # Creamos un indicador en Streamlit para el precio
            st.markdown(f"#### 💰 Precio de lista promedio")
            col_p_ars, col_p_ipc, col_p_usd = st.columns(3)

            valor1 = format_ars_value(precio_promedio_ars)
            valor2 = format_ars_value(precio_promedio_ipc)
            valor3 = format_ars_value(precio_promedio_usd)

            with col_p_ars:
                st.metric(label="Precio (ARS)", value=f'${valor1}', border=True)
                # st.write(precio_promedio_ars)
            with col_p_ipc:
                st.metric(label="Precio IPC (ARS)", value=f"${valor2}",border=True)
            with col_p_usd:
                st.metric(label="Precio (USD)", value=f"U$D {valor3}",border=True)
            
        else:
            st.warning(":warning: No se encontraron datos de **Precios** para la selección actual.")
            precio_promedio_ars = None

        st.write('')
        # st.subheader('', divider='grey')

        # Renombrar la columna de df_tabla2 para que coincida con df_tabla3
        df_tabla2_renombrado = df_tabla2.rename(columns={'año_mes_fecha_pago': 'fecha'})

        df_comparacion = pd.merge(
            df_tabla3,                 # Precio (Tabla Izquierda)
            df_tabla2_renombrado,      # Pago (Tabla Derecha)
            on='fecha',                # Columna clave de unión
            how='inner',               # Solo filas que existen en ambos
            suffixes=('_precio', '_pago') # Sufijos para diferenciar columnas duplicadas
        )

        columnas_presentacion = [
            'fecha',
            'precio_total', 
            'precio_total_ipc',
            'precio_total_usd',
            'monto_transaccion',
            'pago_ipc',
            'pago_usd'
        ]

        df_comparacion_final = df_comparacion[columnas_presentacion].copy()

        renombre_columnas = {
            'fecha': 'Fecha',
            'precio_total': 'Precio Total (ARS)',
            'precio_total_ipc': 'Precio IPC (ARS)',
            'precio_total_usd': 'Precio Total (USD)',
            'monto_transaccion': 'Pago Total (ARS)',
            'pago_ipc': 'Pago IPC (ARS)',
            'pago_usd': 'Pago Total (USD)'
        }

        df_comparacion_final.rename(columns=renombre_columnas, inplace=True)

        columnas_numericas_ars = [col for col in df_comparacion_final.columns if 'ARS' in col]

        for col in columnas_numericas_ars:
            # Usar el locale para formatear sin decimales
            df_comparacion_final[col] = df_comparacion_final[col].apply(
                lambda x: locale.format_string("$ %.0f", x, grouping=True)
            )

        columnas_numericas_usd = [col for col in df_comparacion_final.columns if 'USD' in col]
        for col in columnas_numericas_usd:
            # Usar el locale para formatear con el prefijo U$D
            df_comparacion_final[col] = df_comparacion_final[col].apply(
                lambda x: locale.format_string("$ %.0f", x, grouping=True)
            )

        col1, col2, col3 = st.columns(3)
        if  not df_tabla2.empty:
            with col1:
                st.markdown("**VALORES ARS**")
                
                # Extraer los valores de la única fila
                precio_ars = df_comparacion_final['Precio Total (ARS)'].iloc[0]
                pago_ars = df_comparacion_final['Pago Total (ARS)'].iloc[0]
                
                # Mostrar como métricas simples
                st.text(f"Pago: {pago_ars}")
                st.text(f"Precio: {precio_ars}")
                st.text("---")

                pago_ars_num = float(pago_ars.replace('$', '').replace('.', '').replace(',', '').strip())
                precio_ars_num = float(precio_ars.replace('$', '').replace('.', '').replace(',', '').strip())
                
                diff_ars_numerico = pago_ars_num - precio_ars_num 
                if precio_ars_num != 0:
                    delta_ars_percent = (diff_ars_numerico / precio_ars_num) * 100
                else:
                    # Manejo de división por cero
                    delta_ars_percent = 0

                valor_final_display = format_ars_delta(diff_ars_numerico)

                st.metric(label="Diferencia en pagos ARS", 
                          value=valor_final_display, 
                          delta=f"{delta_ars_percent:+.2f} %",
                          delta_color='inverse')

                
            # 2. Columna IPC (Ajustado por IPC)
            with col2:
                st.markdown("**VALORES IPC**")
                
                precio_ipc = df_comparacion_final['Precio IPC (ARS)'].iloc[0]
                pago_ipc = df_comparacion_final['Pago IPC (ARS)'].iloc[0]
                
                st.text(f"Pago: {pago_ipc}")
                st.text(f"Precio: {precio_ipc}")
                st.text("---")

                pago_ipc_num = float(pago_ipc.replace('$', '').replace('.', '').replace(',', '').strip())
                precio_ipc_num = float(precio_ipc.replace('$', '').replace('.', '').replace(',', '').strip())
                diff_ipc = pago_ipc_num - precio_ipc_num
                if precio_ipc_num != 0:
                    delta_ipc_percent = (diff_ipc / precio_ipc_num) * 100
                else:
                    # Manejo de división por cero
                    delta_ipc_percent = 0

                valor_final_ipc = format_ars_delta(diff_ipc)

                st.metric(label="Diferencia en pagos IPC", 
                          value=valor_final_ipc, 
                          delta=f"{delta_ipc_percent:+.2f} %",
                          delta_color='inverse')
                
            # 3. Columna USD
            with col3:
                st.markdown("**VALORES USD**")
                
                precio_usd = df_comparacion_final['Precio Total (USD)'].iloc[0]
                pago_usd = df_comparacion_final['Pago Total (USD)'].iloc[0]

                st.text(f"Pago: {pago_usd}")           
                st.text(f"Precio: {precio_usd}")
                st.text("---")
                pago_usd_num = float(pago_usd.replace('$', '').replace('.', '').replace(',', '').strip())
                precio_usd_num = float(precio_usd.replace('$', '').replace('.', '').replace(',', '').strip())

                diff_usd = pago_usd_num - precio_usd_num
                # Evitar división por cero
                if precio_usd_num != 0:
                    delta_usd_percent = (diff_usd / precio_usd_num) * 100
                else:
                    delta_usd_percent = 0

                valor_final_usd = format_ars_delta(diff_usd)

                st.metric(label="Diferencia en pagos USD", 
                          value=valor_final_usd, 
                          delta=f"{delta_usd_percent:+.2f} %",
                          delta_color='inverse')


            st.subheader('', divider='grey')
            # # 4. Mostrar el DataFrame debajo para detalles (si el usuario lo requiere)
            # st.markdown("#### **Datos Crudos Filtrados**")
            # st.dataframe(df_comparacion_final, hide_index=True, use_container_width=True)
        else:
            st.warning(":warning: No se encontraron datos para mostrar la comparación detallada.")

        # VENIR ACA


        def create_historical_comparison_plot(df_pagos, df_precios, pago_col, precio_col, y_label, title):

            df_pagos_agg = df_pagos.copy()

            # filtro por tipo_cristal
            df_pagos_agg = df_pagos_agg[(df_pagos_agg['tipo_cristal'] == selected_cristal) &
                                        (df_pagos_agg['marca_vehiculo'] == selected_marcas_comp)]
                            
            df_pagos_agg['año_mes_fecha_pago'] = pd.to_datetime(df_pagos_agg['año_mes_fecha_pago'])
            df_pagos_agg['fecha'] = df_pagos_agg['año_mes_fecha_pago'].dt.strftime('%Y-%m')
            
            # Agrupar por fecha y obtener el promedio de la columna de pago
            df_pagos_agg = df_pagos_agg.groupby('fecha').agg({pago_col: 'mean'}).reset_index()
            df_pagos_agg.rename(columns={pago_col: 'Pago Promedio'}, inplace=True)

            df_precios_agg = df_precios.copy()
            df_precios_agg = df_precios_agg[(df_precios_agg['tipo_cristal'] == selected_cristal) &
                                            (df_precios_agg['marca_vehiculo'] == selected_marcas_comp)]
            df_precios_agg['fecha'] = pd.to_datetime(df_precios_agg['fecha'])
            df_precios_agg['fecha'] = df_precios_agg['fecha'].dt.strftime('%Y-%m')

            # Agrupar por fecha y obtener el promedio de la columna de precio
            df_precios_agg = df_precios_agg.groupby('fecha').agg({precio_col: 'mean'}).reset_index()
            df_precios_agg.rename(columns={precio_col: 'Precio Promedio'}, inplace=True)

            # 3. Unión de DataFrames
            # Merge con la clave 'fecha' (que ahora son ambas cadenas 'YYYY-MM')
            df_comparacion = pd.merge(
                df_precios_agg,
                df_pagos_agg,
                on='fecha',
                how='outer'
            ).sort_values('fecha')

            df_long = pd.melt(
                df_comparacion, 
                id_vars=['fecha'], 
                value_vars=['Precio Promedio', 'Pago Promedio'],
                var_name='Métrica', 
                value_name='Monto'
            ).dropna(subset=['Monto']) # Eliminar NaNs para que las líneas no se rompan

            fig = px.line(
                df_long,
                x='fecha',
                y='Monto',
                color='Métrica',
                title=title,
                labels={'fecha': '', 'Monto': y_label, 'Métrica': ''},
                height=550
            )

            # Ajustes visuales
            fig.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis_tickprefix=y_label.split('(')[0].strip(), # Añadir símbolo de moneda ($, U$D, etc.)
                title=dict(
                    font=dict(size=20, family="Arial")   
            ))

            fig.update_yaxes(
                title_text=y_label, # Mantenemos la etiqueta limpia
                tickprefix="",      # Elimina cualquier prefijo de texto no deseado
                tickformat=".3s"    # Esto usa la notación SI, mostrando 340k en lugar de 340000
            )
            fig.update_traces(mode='lines+markers', line=dict(width=3))
            
            return fig
                
        # --- Definición de las tres llamadas a la función ---

        # 1. Gráfico TOTAL (ARS)
        fig_total = create_historical_comparison_plot(
            df_pagos_cristal,
            df_cristal_copy,
            pago_col='monto_transaccion',
            precio_col='precio_total',
            y_label='Monto (ARS)',
            title='Evolución: Monto Total de Pago vs. Precio de Lista (ARS)'
        )

        # 2. Gráfico IPC (ARS Ajustado)
        fig_ipc = create_historical_comparison_plot(
            df_pagos_cristal,
            df_cristal_copy,
            pago_col='pago_ipc',
            precio_col='precio_total_ipc',
            y_label='Monto IPC (ARS)',
            title='Evolución: Monto de Pago vs. Precio de Lista (Ajustado por IPC)'
        )

        # 3. Gráfico USD (Dólar)
        fig_usd = create_historical_comparison_plot(
            df_pagos_cristal,
            df_cristal_copy,
            pago_col='pago_usd',
            precio_col='precio_total_usd',
            y_label='Monto (USD)',
            title='Evolución: Monto de Pago vs. Precio de Lista (USD)'
        )

        tab1, tab2, tab3 = st.tabs(["Total (ARS)", "IPC (ARS)", "USD"])
        with tab1:
            st.plotly_chart(fig_total, use_container_width=True)

        with tab2:
            st.plotly_chart(fig_ipc, use_container_width=True)

        with tab3:
            st.plotly_chart(fig_usd, use_container_width=True)



# # ==== DATA CRUDA ===========================================
#         cols = ['fecha_de_pago', 'nro_siniestro_gw', 'cobertura_principal', 'tipo_cristal',
#                 'año_mes_fecha_pago', 'marca_vehiculo', 'modelo_vehiculo','provincia', 'monto_transaccion',
#                 'ipc_x_fecha_de_pago', 'pago_ipc', 'usd_blue_fp', 'pago_usd', 'zona']
        
#         df_to_show = df_pagos_cristal[cols]

#         # if not selected_cristal:
#         #     st.dataframe(df_to_show[df_to_show['marca_vehiculo'].isin(selected_marcas)].drop('año_mes_fecha_pago',axis=1), use_container_width=True)
#         # else:
#         st.markdown("##### Data Cruda (pagos)")
#         df_pagos_cristal_filtered = df_to_show.drop(columns=['cobertura_principal','año_mes_fecha_pago'])[
#             (df_to_show['tipo_cristal'] == selected_cristal) &
#             (df_to_show['marca_vehiculo'].isin(selected_marcas))
#             # & (df_to_show['zona'].isin(selected_zone))
#         ].copy()
#         st.dataframe(df_pagos_cristal_filtered, use_container_width=True)


# ==========================================================================
# ---- Análisis ORION/CESVI ------------------------------------------------
# ==========================================================================
    elif current_analysis == opcion_3:
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
                #title='', agrego titulo con subheader
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

            # st.markdown("---")
        
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

            tab1, tab2 = st.tabs(["Evolutivo CM", "Variación vs IPC"])
            with tab1:
                st.plotly_chart(fig5, use_container_width=True)
            with tab2:
                st.plotly_chart(fig5_ipc, use_container_width=True)

            st.subheader('', divider='grey')
    # ==========================================================================

            # GRAFICO 2: evolución costo repuestos por tipo repuesto
            st.subheader('2. Costo de piezas prom. histórico por Tipo Repuesto')
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
                st.markdown('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
                st.markdown('Total marcas: 44')
                st.markdown("---")
    # ==========================================================================

            # GRAFICO 3: evolución costo repuestos por marca autos
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
                st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
                st.text('Total marcas: 26')
                st.markdown("---")
    # ==========================================================================

            # GRAFICO 4: evolución costo repuestos por marca camiones
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
                st.dataframe(df_cm_mo_cleas, hide_index=True,)
                    
            # quito camion_cleas_si del df resumen por poca cantidad de datos
            df_cm_mo_cleas = df_cm_mo_cleas[df_cm_mo_cleas['tva'] != 'camion_cleas_si']
            fig14 = create_plot_orion(df_cm_mo_cleas, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio', 45)

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
                st.text('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
                st.text('Total marcas: 44' )
                st.markdown("---")
    # ==========================================================================

            # gráfico 3: evolución costo repuestos por marca autos IPC
            st.subheader('3. Costo de piezas prom. por Marca (autos) - Ajust. por IPC')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                st.dataframe(df_rtos_marca_mes[['marca','año_mes','cant_ocompra','cant_piezas_total',
                                        'costo_prom_ipc','var_costo_prom_ipc','monto_ipc']], hide_index=True,)

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
                st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
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

            # GRAFICO 5: evolución costo mano de obra por tva y tipo de mano de obra IPC
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
            st.plotly_chart(fig12, use_container_width=True)
            st.markdown("---")
    # ==========================================================================

            # GRAFICO 6: evolución costo mano de obra cleas si vs cleas no IPC
            st.subheader('6. Comparativa variación M.O - CLEAS SI vs CLEAS NO - Ajust. por IPC')

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=":material/query_stats:"):
                st.dataframe(df_cm_mo_cleas, hide_index=True,)

            # quito camion_cleas_si del df resumen por poca cantidad de datos
            df_cm_mo_cleas = df_cm_mo_cleas[df_cm_mo_cleas['tva'] != 'camion_cleas_si']
            fig15 = create_plot_orion(df_cm_mo_cleas, 'valor_ipc', 'tva','tipo_costo', 'Costo Promedio ajust. por IPC')
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
                st.text('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
                st.text('Total marcas: 44' )
                st.markdown("---")
    # ==========================================================================

            # gráfico 3: evolución costo repuestos por marca autos USD
            st.subheader('3. Costo de piezas prom. histórico por Marca (autos) en USD')

            # muestro el dataset
            with st.expander("Ver tabla de datos",icon=":material/query_stats:"):
                st.dataframe(df_rtos_marca_mes[['marca','año_mes','cant_ocompra','cant_piezas_total', 'usd_blue',
                                        'costo_prom_usd','var_costo_prom_usd','monto_usd']], hide_index=True,)

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
                st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
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
            st.plotly_chart(fig13, use_container_width=True)
            st.markdown("---")
    # ==========================================================================

            # gráfico 6: evolución costo mano de obra cleas si vs cleas no USD
            st.subheader('6. Comparativa variación M.O en USD - CLEAS SI vs CLEAS NO')

            # muestro el dataset
            with st.expander("Ver tabla de datos",icon=":material/query_stats:"):
                st.dataframe(df_cm_mo_cleas, hide_index=True,)

            # quito camion_cleas_si del df resumen por poca cantidad de datos
            df_cm_mo_cleas = df_cm_mo_cleas[df_cm_mo_cleas['tva'] != 'camion_cleas_si']
            fig16 = create_plot_orion(df_cm_mo_cleas, 'valor_usd', 'tva','tipo_costo', 'Costo Promedio (USD)')
            st.plotly_chart(fig16, use_container_width=True)

        '''Se descarta línea del gráfico de 'camion_cleas_si' por poca cantidad de datos'''
# ==========================================================================
# ---- Análisis por PROVINCIA ----------------------------------------------
# ==========================================================================
    elif current_analysis == opcion_4:
        st.title('Análisis Coste Medio por Provincia')     
        st.markdown("---")   
        st.header('Coste Medio de repuestos por provincia')
        st.markdown("#### _Fuente de datos: Orion/Cesvi_")
        

        def create_map_chart(df, selected_coverable, color, selected_fecha):
            df_cm_filtered = df[(df['coverable'] == selected_coverable) &
                                (df['año'] == selected_fecha)]

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
        
        with st.expander("Ver data cruda",icon=":material/query_stats:"):
            st.markdown("#### Data Cruda")
            # Para mostrar los datos crudos filtrados (opcional, ajusta tu lógica de datos)
            df_cm_prov_orion_raw = df_cm_prov_orion[(df_cm_prov_orion['coverable'] == selected_coverable_map) &
                                                        (df_cm_prov_orion['año'] == selected_fecha)]
            st.dataframe(df_cm_prov_orion_raw, use_container_width=True)
    # ==========================================================================

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

        with st.expander("Ver data cruda",icon=":material/query_stats:"):
            st.markdown("#### Data Cruda")
            # Para mostrar los datos crudos filtrados (opcional, ajusta tu lógica de datos)
            df_cm_filtered_raw = df_cm_prov[(df_cm_prov['coverable'] == selected_coverable_map) &
                                                            (df_cm_prov['año'] == selected_fecha)]
            st.dataframe(df_cm_filtered_raw, use_container_width=True)   

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

            fig.update_layout(
                legend_title_text=leg_title_text,
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
            y_cols_hist = ['grupo_cesvi', 'grupo_sls', 'la_segunda', 'san_cristobal', 'sancor']
            
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
                st.dataframe(df_tot_reparacion[['anio_mes', 'grupo_cesvi', 'grupo_sls', 'la_segunda', 'san_cristobal',
                                        'sancor', 'ipc','ipc_empalme_ipim','var_ipc', 'var_%_grupo_cesvi', 'var_%_grupo_sls', 'var_%_la_segunda',
                                        'var_%_san_cristobal', 'var_%_sancor']], hide_index=True,)

            y_cols_var = ['var_%_grupo_cesvi', 'var_%_grupo_sls', 'var_%_la_segunda', 'var_%_san_cristobal', 'var_%_sancor']

            fig_3_ipc = create_plot_mo(df_tot_reparacion, y_cols_var, None, None, 'Variación (base 1)')
            fig_3_ipc.add_trace(go.Scatter(
                x=df_tot_reparacion['anio_mes'],
                y=df_tot_reparacion['var_ipc'],
                name='var_ipc', 
                mode='lines',
                line=dict(color='white', dash='dot'),
            ))
            fig_3_ipc.update_layout(legend_title_text='')

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
                st.dataframe(df_costo_hora[['anio_mes', 'grupo_cesvi', 'grupo_sls', 'la_segunda', 'san_cristobal',
                                        'sancor', 'ipc','ipc_empalme_ipim','var_ipc', 'var_%_grupo_cesvi', 'var_%_grupo_sls', 'var_%_la_segunda',
                                        'var_%_san_cristobal', 'var_%_sancor']], hide_index=True,)

            fig_5_ipc = create_plot_mo(df_costo_hora, y_cols_var, None, None, 'Variación (base 1)')
            fig_5_ipc.add_trace(go.Scatter(
                x=df_costo_hora['anio_mes'],
                y=df_costo_hora['var_ipc'],
                name='var_ipc', 
                mode='lines',
                line=dict(color='white', dash='dot'),
            ))
            fig_5_ipc.update_layout(legend_title_text='')

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
                st.dataframe(df_peritaciones[['anio_mes', 'grupo_cesvi', 'grupo_sls', 'la_segunda', 'san_cristobal', 'sancor']], hide_index=True, width=1000,)
    # ==========================================================================

            st.subheader('▫️ % Variación mensual de cantidad de Peritaciones')
            y_var=['var_%_grupo_cesvi', 'var_%_grupo_sls', 'var_%_la_segunda', 'var_%_san_cristobal', 'var_%_sancor']
            fig_5 = create_plot_mo(df_peritaciones, y_var, None, None, '% variación', leg_title_text='')
            st.plotly_chart(fig_5, use_container_width=True)

            # muestro el dataset
            with st.expander("Ver tabla de datos",):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_peritaciones[['anio_mes', 'part_grupo_sls_vs_cesvi', 'part_la_segunda_vs_cesvi', 'part_san_cristobal_vs_cesvi', 'part_sancor_vs_cesvi']], 
                            hide_index=True, width=1000,)
    # ==========================================================================
    #             
            st.subheader('▫️ % Participacion respecto a Grupo Cesvi')
            y_cols_part=['part_grupo_sls_vs_cesvi', 'part_sancor_vs_cesvi', 'part_la_segunda_vs_cesvi', 'part_san_cristobal_vs_cesvi', ]
            fig_6 = create_plot_mo(df_peritaciones, y_cols_part, None, None, '% participacion', leg_title_text='')
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
                st.dataframe(df_tot_reparacion[['anio_mes','ipc','grupo_cesvi_ipc', 'grupo_sls_ipc', 'la_segunda_ipc', 'san_cristobal_ipc', 'sancor_ipc']], 
                            hide_index=True, width=1000,)

            st.subheader('', divider='grey') 
        
            st.subheader('Evolución Costo Hora de Mano de Obra - ajust. por IPC')
            fig_5 = create_plot_mo(df_costo_hora, y_cols_ipc, None, None, 'Costo hora')
            st.plotly_chart(fig_5, use_container_width=True)

            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=':material/query_stats:'):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_costo_hora[['anio_mes','ipc','grupo_cesvi_ipc', 'grupo_sls_ipc', 'la_segunda_ipc', 'san_cristobal_ipc', 'sancor_ipc']], hide_index=True,)

    # ----- GRAFICOS USD --------------------------------------------------
        if st.session_state['selected_variation_type_2'] == "USD":
            y_cols_usd = ['grupo_cesvi_usd', 'grupo_sls_usd', 'la_segunda_usd', 'san_cristobal_usd', 'sancor_usd']
            
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
                st.dataframe(df_tot_reparacion[['anio_mes','usd_blue','grupo_cesvi_usd', 'grupo_sls_usd', 'la_segunda_usd', 'san_cristobal_usd', 'sancor_usd']], 
                            hide_index=True, width=1000,)
            
            st.subheader('', divider='grey') 

            st.subheader('Evolución Costo Hora de Mano de Obra - en USD')
            fig_5 = create_plot_mo(df_costo_hora, y_cols_usd, None, None, 'Costo hora')
            st.plotly_chart(fig_5, use_container_width=True)
            # muestro el dataset
            with st.expander("Ver tabla de datos", icon=':material/query_stats:'):
                # st.subheader("Tabla de Datos de Ejemplo")
                st.dataframe(df_costo_hora[['anio_mes','usd_blue','grupo_cesvi_usd', 'grupo_sls_usd', 'la_segunda_usd', 'san_cristobal_usd', 'sancor_usd']], 
                            hide_index=True, width=1000,)
# ==========================================================================

    elif current_analysis == opcion_6:
        st.title('Evolución de monto de pagos de Robo y Hurto de ruedas en La Segunda')    
        st.markdown("---")



