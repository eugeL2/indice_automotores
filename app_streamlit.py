import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go # Necesario para go.Figure en caso de datos vacíos

st.set_page_config(layout="wide") # Ancho completo de la página

try:
    # dfs de cristales pilkington
    df_mo_long_ipc = pd.read_csv('data/df_mo_long_ipc.csv')
    df_mo_long = pd.read_csv('data/df_mo_long.csv')
    df_long_ipc = pd.read_csv('data/df_long_ipc.csv')
    df_long = pd.read_csv('data/df_long.csv')

    # dfs de repuestos orion/cesvi
    df_tipo_rep = pd.read_csv('data/df_tipo_rep.csv')
    df_rep_tv = pd.read_csv('data/df_rep_tv.csv')

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
    df_rtos_marca_mes = pd.read_csv('data/df_rtos_marca_mes.csv')
    df_marcas_camiones = pd.read_csv('data/camion_marcas.csv')
    df_rtos_marca_mes_cam = pd.read_csv('data/df_rtos_marca_mes_camiones.csv')

except FileNotFoundError:
    st.error("Error: No se encuentran los archivos CSV.")
    # la app se detiene si no encuentra los archivos
    st.stop() 

for df_temp in [df_long, df_mo_long, df_long_ipc, df_mo_long_ipc, df_tipo_rep]:
    if 'fecha' in df_temp.columns:
        df_temp['fecha'] = pd.to_datetime(df_temp['fecha'])
    if 'año_mes' in df_temp.columns:
        df_temp['año_mes'] = pd.to_datetime(df_temp['año_mes'])
    if 'tipo_cristal' in df_temp.columns:
        df_temp['tipo_cristal'] = df_temp['tipo_cristal'].astype(str).str.replace('_', ' ').str.title()
    if 'marca' in df_temp.columns:
        df_temp['marca'] = df_temp['marca'].astype(str)
    if 'zona' in df_temp.columns:
        df_temp['zona'] = df_temp['zona'].astype(str)
    if 'tipo_repuesto' in df_temp.columns:
        df_temp['tipo_repuesto'] = df_temp['tipo_repuesto'].astype(str).str.replace('_', ' ').str.title()

st.markdown("### Seleccionar Fuente de Análisis")
selected_analysis = st.selectbox(
    'Seleccionar Análisis:',
    options=["PILKINGTON", 
             "ORION/CESVI"],
    index=0,
    label_visibility ='collapsed'
)
st.markdown("---")

if 'show_pie_chart' not in st.session_state:
    st.session_state.show_pie_chart = False

if 'show_pie_chart_2' not in st.session_state:
    st.session_state.show_pie_chart_2 = False

if selected_analysis == "PILKINGTON":
    st.title("Variación de Precios de Cristales y Mano de obra por Marca y Zona")
    st.markdown("#### _Fuente de datos: Listas de precios de Pilkington_")
    st.markdown("---")

    # Dropdown de Zona (barra lateral)
    available_zones = sorted(df_long['zona'].unique().tolist())
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

    def create_plot_pkt(df_source, y_col, y_label):
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

        # gráfico Plotly 
        fig = px.line(
            df_filtered,
            x='fecha',
            y=y_col,
            color='marca', # Un color para cada marca
            line_group='marca',
            facet_col='tipo_cristal', # Subplots por tipo de cristal
            #title='', agrego titulo con subheader
            labels={'fecha': '', y_col: y_label, 'marca': 'Marca', 'tipo_cristal': 'Tipo de Cristal'}
        )

        # Ajustes del gráfico
        fig.update_layout(
            height=400, # Altura del subplot individual
            legend_title_text='Marca',
            font=dict(family="Arial", size=15),
            #title_font_size=12,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        # Ajustar el título de las facetas para que sean más legibles
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        return fig

    # graficos de cristales (apilados)
    st.subheader('1. Precios de Material históricos (Sin IVA)')
    fig1 = create_plot_pkt(df_long, 'precio', 'Precio Sin IVA')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader('2. Costo de Instalación histórico (Sin IVA)')
    fig2 = create_plot_pkt(df_mo_long, 'instalacion', 'Costo de Instalación')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader('3. Precios de Material (Ajustados por IPC)')
    fig3 = create_plot_pkt(df_long_ipc, 'precio_ipc', 'Precio (IPC)')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader('4. Costo de Instalación (Ajustados por IPC)')
    fig4 = create_plot_pkt(df_mo_long_ipc, 'instalacion_ipc', 'Costo de Instalación (IPC)')
    st.plotly_chart(fig4, use_container_width=True)

elif selected_analysis == "ORION/CESVI":
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
    
    def create_plot_orion(df, y_col, color, facet_col, y_label):       
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
            margin=dict(t=50, b=0, l=0, r=0)
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        return fig
    
    # grafico de torta de df_marcas_autos
    def create_pie_chart(df):
        fig = px.pie(
        df,
        values='cant_ocompra',
        names='marca_agrupada',
        hover_data=['cant_ocompra'],
        color_discrete_sequence=px.colors.qualitative.G10,
        labels={'cant_ocompra': 'Cant. Órdenes', 'marca_agrupada': 'Marca'},
        hole=0.3 # Agrega un agujero al centro para un estilo de "donut chart"
        )

        fig.update_traces(
            # textposition='inside',
            textinfo='percent+label',
            insidetextfont=dict(size=12, color='black', family='Arial')
        )
        fig.update_layout(
            font=dict(family="Arial", size=12, color="black"),
            showlegend=True
        )

        return fig   
    
    # gráficos históricos
    if st.session_state['selected_variation_type'] == "Histórico":

        st.subheader('1. Costo de piezas prom. histórico por TVA')
        fig5 = create_plot_orion(df_rep_tv, 'costo_pieza_prom_hist', 'tva', None,'Costo Promedio')
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("---")

        st.subheader('2. Costo de piezas prom. histórico por Tipo Repuesto')
        fig6 = create_plot_orion(df_tipo_rep, 'costo_pieza_prom', 'tipo_repuesto', None,'Costo Promedio')
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("---")

        if st.button("Mostrar/Ocultar Distribución de Marcas Autos"):
            st.session_state.show_pie_chart = not st.session_state.show_pie_chart
        
        if st.session_state.show_pie_chart:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_autos)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
            st.markdown('Total marcas: 44' )
            st.markdown("---")

        st.subheader('3. Costo de piezas prom. histórico por Marca (autos)')
        fig17 = create_plot_orion(df_rtos_marca_mes, 'costo_pieza_prom_hist', 'marca', None, 'Costo Promedio')
        st.plotly_chart(fig17, use_container_width=True)
        st.markdown("---")

        if st.button("Mostrar/Ocultar Distribución de Marcas Camiones"):
            st.session_state.show_pie_chart_2 = not st.session_state.show_pie_chart_2
        
        if st.session_state.show_pie_chart_2:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_camiones)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
            st.text('Total marcas: 26')
            st.markdown("---")

        st.subheader('4. Costo de piezas prom. histórico por Marca (camiones)')
        fig20 = create_plot_orion(df_rtos_marca_mes_cam, 'costo_pieza_prom_hist', 'marca', None, 'Costo Promedio')
        st.plotly_chart(fig20, use_container_width=True)
        st.markdown("---")      

        st.subheader('5. Costo de mano de obra prom. histórico por Tipo de M.O y TVA')
        fig11 = create_plot_orion(df_cm_mo_hist, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio')
        st.plotly_chart(fig11, use_container_width=True)
        st.markdown("---")

        st.subheader('6. Comparativa variación M.O - CLEAS SI vs CLEAS NO')
        fig14 = create_plot_orion(df_cm_mo_hist_cleas, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio')
        st.plotly_chart(fig14, use_container_width=True)
        
    # gráficos ajustados por IPC
    elif st.session_state['selected_variation_type'] == "IPC":

        st.subheader('1. Evolución del costo prom. por TVA - Ajust. por IPC')
        fig7 = create_plot_orion(df_rep_tv, 'costo_prom_ipc', 'tva', None, 'Costo Promedio Ajust. por IPC')
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("---")
    
        st.subheader('2. Evolución del costo prom. por Tipo Repuesto - Ajust. por IPC')
        fig8 = create_plot_orion(df_tipo_rep, 'costo_prom_ipc', 'tipo_repuesto', None,'Costo Promedio ajust. por IPC')
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("---")

        if st.button("Mostrar/Ocultar Distribución de Marcas Autos"):
            st.session_state.show_pie_chart = not st.session_state.show_pie_chart
        
        if st.session_state.show_pie_chart:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_autos)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.text('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
            st.text('Total marcas: 44' )
            st.markdown("---")

        st.subheader('3. Costo de piezas prom. por Marca (autos) - Ajust. por IPC')
        fig18 = create_plot_orion(df_rtos_marca_mes, 'costo_prom_ipc', 'marca', None, 'Costo Promedio')
        st.plotly_chart(fig18, use_container_width=True)
        st.markdown("---")

        if st.button("Mostrar/Ocultar Distribución de Marcas Camiones"):
            st.session_state.show_pie_chart_2 = not st.session_state.show_pie_chart_2
        
        if st.session_state.show_pie_chart_2:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_camiones)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
            st.text('Total marcas: 26')
            st.markdown("---")

        st.subheader('4. Costo de piezas prom. por Marca (camiones) - Ajust. por IPC')
        fig21 = create_plot_orion(df_rtos_marca_mes_cam, 'costo_prom_ipc', 'marca', None, 'Costo Promedio')
        st.plotly_chart(fig21, use_container_width=True)
        st.markdown("---")    

        st.subheader('5. Evolución del costo de mano de obra prom. por Tipo de M.O y TVA - Ajust. por IPC')
        fig12 = create_plot_orion(df_cm_mo_ipc, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio ajust. por IPC')
        st.plotly_chart(fig12, use_container_width=True)
        st.markdown("---")

        st.subheader('6. Comparativa variación M.O - CLEAS SI vs CLEAS NO - Ajust. por IPC')
        fig15 = create_plot_orion(df_cm_mo_ipc_cleas, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio ajust. por IPC')
        st.plotly_chart(fig15, use_container_width=True)

    # gráficos en USD
    elif st.session_state['selected_variation_type'] == "USD":

        st.subheader('1. Evolución del costo prom. por TVA en USD')
        fig9 = create_plot_orion(df_rep_tv, 'costo_prom_usd', 'tva', None, 'Costo Promedio (USD)')
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown("---")

        st.subheader('2. Evolución del costo prom. por Tipo Repuesto en USD')
        fig10 = create_plot_orion(df_tipo_rep, 'costo_prom_usd', 'tipo_repuesto', None, 'Costo Promedio (USD)')
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown("---")

        if st.button("Mostrar/Ocultar Distribución de Marcas Autos"):
            st.session_state.show_pie_chart = not st.session_state.show_pie_chart
        
        if st.session_state.show_pie_chart:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_autos)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.text('Total ord. compra autos (ene23-jul25): ' + str(df_marcas_autos['cant_ocompra'].sum()))
            st.text('Total marcas: 44' )
            st.markdown("---")

        st.subheader('3. Costo de piezas prom. histórico por Marca (autos) en USD')
        fig19 = create_plot_orion(df_rtos_marca_mes, 'costo_prom_usd', 'marca', None, 'Costo Promedio (USD)')
        st.plotly_chart(fig19, use_container_width=True)
        st.markdown("---")

        if st.button("Mostrar/Ocultar Distribución de Marcas Camiones"):
            st.session_state.show_pie_chart_2 = not st.session_state.show_pie_chart_2
        
        if st.session_state.show_pie_chart_2:
            st.subheader('Distribución de Órdenes de Compra por Marca')
            fig_pie = create_pie_chart(df_marcas_camiones)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.text('Total ord. compra camiones (ene23-jul25): ' + str(df_marcas_camiones['cant_ocompra'].sum()))
            st.text('Total marcas: 26')
            st.markdown("---")

        st.subheader('4. Costo de piezas prom. por Marca (camiones) en USD')
        fig22 = create_plot_orion(df_rtos_marca_mes_cam, 'costo_prom_usd', 'marca', None, 'Costo Promedio (USD)')
        st.plotly_chart(fig22, use_container_width=True)
        st.markdown("---") 

        st.subheader('5. Evolución del costo de Mano de Obra prom. por Tipo de M.O y TVA en USD')
        fig13 = create_plot_orion(df_cm_mo_usd, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio (USD)')
        st.plotly_chart(fig13, use_container_width=True)
        st.markdown("---")

        st.subheader('6. Comparativa variación M.O en USD - CLEAS SI vs CLEAS NO')
        fig16 = create_plot_orion(df_cm_mo_usd_cleas, 'valor_costo', 'tva','tipo_costo', 'Costo Promedio (USD)')
        st.plotly_chart(fig16, use_container_width=True)
