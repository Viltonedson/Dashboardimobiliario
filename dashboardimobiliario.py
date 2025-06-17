import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard Imobiliário - Recife",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para carregar e processar os dados
@st.cache_data
def load_data():
    # Simulando o carregamento dos dados do CSV
    data = {
        'data': pd.date_range('2011-01-01', '2021-12-01', freq='MS'),
        'ipca': np.random.normal(0.6, 0.3, 132),
        'selic': np.random.normal(0.7, 0.3, 132),
        'cdi': np.random.normal(0.7, 0.3, 132),
        'fipezap_rec': np.random.normal(180, 20, 132),
        'ano': [d.year for d in pd.date_range('2011-01-01', '2021-12-01', freq='MS')],
        'pib_br': np.random.normal(500000, 100000, 132),
        'pib_pe': np.random.normal(170000000, 30000000, 132),
        'pib_rec': np.random.normal(50000000, 5000000, 132),
        'pop_rec': np.random.normal(1600000, 50000, 132),
        'renda_per_capita_rec': np.random.normal(30000, 3000, 132)
    }
    
    df = pd.DataFrame(data)
    df['data'] = pd.to_datetime(df['data'])
    df['variacao_fipezap'] = df['fipezap_rec'].pct_change() * 100
    df['mes'] = df['data'].dt.month
    df['trimestre'] = df['data'].dt.quarter
    
    return df

# Função para criar modelo de previsão
def create_prediction_model(df):
    features = ['ipca', 'selic', 'cdi', 'pib_br', 'pib_pe', 'pib_rec', 'renda_per_capita_rec', 'mes', 'trimestre']
    X = df[features].fillna(df[features].mean())
    y = df['fipezap_rec'].fillna(df['fipezap_rec'].mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, r2, features

# Função para criar mapa de valorização
def create_valuation_map():
    # Dados simulados para diferentes regiões de Recife
    regioes = {
        'região': ['Boa Viagem', 'Recife Antigo', 'Casa Forte', 'Graças', 'Espinheiro', 
                  'Aflitos', 'Derby', 'Ilha do Leite', 'Pina', 'Brasília Teimosa'],
        'lat': [-8.113, -8.063, -8.038, -8.043, -8.048, -8.053, -8.058, -8.068, -8.088, -8.098],
        'lon': [-34.896, -34.871, -34.918, -34.893, -34.888, -34.883, -34.878, -34.873, -34.878, -34.886],
        'preco_m2': [8500, 6200, 9200, 7800, 7200, 6800, 5900, 5400, 6100, 4200],
        'variacao_12m': [12.5, 8.3, 15.2, 10.7, 9.4, 7.8, 5.2, 3.9, 6.7, 2.1]
    }
    
    df_map = pd.DataFrame(regioes)
    return df_map

# Título principal
st.title("🏠 Dashboard Imobiliário - Recife")
st.markdown("---")

# Carregar dados
df = load_data()

# Sidebar para filtros
st.sidebar.header("🔧 Configurações")
periodo_inicio = st.sidebar.date_input("Data Início", df['data'].min().date())
periodo_fim = st.sidebar.date_input("Data Fim", df['data'].max().date())

# Filtrar dados por período
df_filtrado = df[(df['data'] >= pd.to_datetime(periodo_inicio)) & 
                 (df['data'] <= pd.to_datetime(periodo_fim))]

# Layout em abas
tab1, tab2, tab3, tab4 = st.tabs(["📈 Previsões", "🗺️ Mapa de Valorização", "🔗 Correlações", "⚙️ Simulador"])

# ABA 1: PREVISÕES
with tab1:
    st.header("Previsão de Valorização Imobiliária")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico histórico e previsão
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_filtrado['data'],
            y=df_filtrado['fipezap_rec'],
            mode='lines',
            name='FIPEZAP Recife (Histórico)',
            line=dict(color='blue')
        ))
        
        # Adicionar linha de tendência
        z = np.polyfit(range(len(df_filtrado)), df_filtrado['fipezap_rec'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df_filtrado['data'],
            y=p(range(len(df_filtrado))),
            mode='lines',
            name='Tendência',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(
            title="Evolução dos Preços Imobiliários",
            xaxis_title="Data",
            yaxis_title="Índice FIPEZAP",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Métricas principais
        variacao_atual = df_filtrado['variacao_fipezap'].iloc[-1] if not df_filtrado['variacao_fipezap'].empty else 0
        preco_atual = df_filtrado['fipezap_rec'].iloc[-1] if not df_filtrado['fipezap_rec'].empty else 0
        
        st.metric("Variação Mensal", f"{variacao_atual:.2f}%")
        st.metric("Índice Atual", f"{preco_atual:.2f}")
        
        # Criar modelo de previsão
        if len(df_filtrado) > 10:
            model, mae, r2, features = create_prediction_model(df_filtrado)
            st.metric("Precisão do Modelo (R²)", f"{r2:.3f}")
            st.metric("Erro Médio Absoluto", f"{mae:.2f}")
    
    # Previsão para próximos meses
    st.subheader("Previsão para os Próximos 6 Meses")
    
    if len(df_filtrado) > 10:
        # Preparar dados para previsão
        last_row = df_filtrado.iloc[-1]
        future_predictions = []
        
        for i in range(6):
            future_data = [
                last_row['ipca'], last_row['selic'], last_row['cdi'],
                last_row['pib_br'], last_row['pib_pe'], last_row['pib_rec'],
                last_row['renda_per_capita_rec'], 
                ((last_row['mes'] + i) % 12) + 1, 
                ((last_row['trimestre'] + i//3 - 1) % 4) + 1
            ]
            pred = model.predict([future_data])[0]
            future_predictions.append(pred)
        
        # Gráfico de previsão
        future_dates = pd.date_range(df_filtrado['data'].max() + pd.DateOffset(months=1), periods=6, freq='MS')
        
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=df_filtrado['data'].tail(12),
            y=df_filtrado['fipezap_rec'].tail(12),
            mode='lines+markers',
            name='Histórico',
            line=dict(color='blue')
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode='lines+markers',
            name='Previsão',
            line=dict(color='red', dash='dash')
        ))
        
        fig_pred.update_layout(
            title="Previsão dos Próximos 6 Meses",
            xaxis_title="Data",
            yaxis_title="Índice FIPEZAP",
            height=400
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)

# ABA 2: MAPA DE VALORIZAÇÃO
with tab2:
    st.header("Mapa de Valorização por Região")
    
    df_map = create_valuation_map()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Mapa de calor
        fig_map = px.scatter_mapbox(
            df_map,
            lat="lat",
            lon="lon",
            size="preco_m2",
            color="variacao_12m",
            hover_name="região",
            hover_data={"preco_m2": True, "variacao_12m": True},
            color_continuous_scale="RdYlGn",
            size_max=30,
            zoom=11,
            height=500
        )
        
        fig_map.update_layout(
            mapbox_style="open-street-map",
            title="Valorização Imobiliária por Região"
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.subheader("Top Regiões")
        
        # Ranking de valorização
        top_regioes = df_map.nlargest(5, 'variacao_12m')[['região', 'variacao_12m', 'preco_m2']]
        
        for idx, row in top_regioes.iterrows():
            st.metric(
                row['região'], 
                f"R$ {row['preco_m2']:,.0f}/m²",
                f"{row['variacao_12m']:+.1f}%"
            )
    
    # Gráfico de barras
    st.subheader("Comparativo de Preços por Região")
    
    fig_bar = px.bar(
        df_map.sort_values('preco_m2', ascending=True),
        x='preco_m2',
        y='região',
        color='variacao_12m',
        color_continuous_scale='RdYlGn',
        title="Preço por m² e Valorização por Região"
    )
    
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

# ABA 3: CORRELAÇÕES
with tab3:
    st.header("Análise de Correlações")
    
    # Matriz de correlação
    numeric_cols = ['fipezap_rec', 'ipca', 'selic', 'cdi', 'pib_br', 'pib_pe', 'pib_rec', 'renda_per_capita_rec']
    corr_matrix = df_filtrado[numeric_cols].corr()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Heatmap de correlação
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title="Matriz de Correlação"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Correlações mais importantes com FIPEZAP
        correlacoes_fipezap = corr_matrix['fipezap_rec'].abs().sort_values(ascending=False)[1:]
        
        fig_bar_corr = px.bar(
            x=correlacoes_fipezap.values,
            y=correlacoes_fipezap.index,
            orientation='h',
            title="Correlações com Índice FIPEZAP"
        )
        fig_bar_corr.update_layout(height=500)
        st.plotly_chart(fig_bar_corr, use_container_width=True)
    
    # Análise detalhada de correlações
    st.subheader("Análise Detalhada")
    
    variavel_selecionada = st.selectbox(
        "Selecione uma variável para análise:",
        options=[col for col in numeric_cols if col != 'fipezap_rec']
    )
    
    if variavel_selecionada:
        fig_scatter = px.scatter(
            df_filtrado,
            x=variavel_selecionada,
            y='fipezap_rec',
            color='ano',
            trendline="ols",
            title=f"Relação entre {variavel_selecionada} e FIPEZAP"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Estatísticas da correlação
        correlation = df_filtrado[variavel_selecionada].corr(df_filtrado['fipezap_rec'])
        st.metric("Coeficiente de Correlação", f"{correlation:.3f}")

# ABA 4: SIMULADOR
with tab4:
    st.header("Simulador de Cenários Econômicos")
    
    st.markdown("Ajuste os parâmetros econômicos para simular o impacto nos preços imobiliários:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Indicadores Macroeconômicos")
        ipca_sim = st.slider("IPCA (%)", -2.0, 5.0, 0.5, 0.1)
        selic_sim = st.slider("SELIC (%)", 0.0, 2.0, 0.7, 0.1)
        cdi_sim = st.slider("CDI (%)", 0.0, 2.0, 0.7, 0.1)
    
    with col2:
        st.subheader("Indicadores Econômicos")
        pib_var = st.slider("Variação PIB BR (%)", -10.0, 10.0, 2.0, 1.0)
        pib_pe_var = st.slider("Variação PIB PE (%)", -10.0, 10.0, 2.0, 1.0)
        renda_var = st.slider("Variação Renda per capita (%)", -10.0, 10.0, 3.0, 1.0)
    
    with col3:
        st.subheader("Parâmetros Temporais")
        mes_sim = st.selectbox("Mês", range(1, 13), index=5)
        trimestre_sim = st.selectbox("Trimestre", range(1, 5), index=1)
    
    # Calcular simulação
    if st.button("🚀 Executar Simulação", type="primary"):
        if len(df_filtrado) > 10:
            # Valores base para simulação
            base_values = df_filtrado.iloc[-1]
            
            # Aplicar variações
            pib_br_sim = base_values['pib_br'] * (1 + pib_var/100)
            pib_pe_sim = base_values['pib_pe'] * (1 + pib_pe_var/100)
            pib_rec_sim = base_values['pib_rec'] * (1 + pib_pe_var/100)  # Assumindo mesma variação
            renda_sim = base_values['renda_per_capita_rec'] * (1 + renda_var/100)
            
            # Dados para simulação
            sim_data = [ipca_sim, selic_sim, cdi_sim, pib_br_sim, pib_pe_sim, 
                       pib_rec_sim, renda_sim, mes_sim, trimestre_sim]
            
            # Fazer previsão
            model, _, _, _ = create_prediction_model(df_filtrado)
            predicao = model.predict([sim_data])[0]
            
            # Comparar com valor atual
            valor_atual = df_filtrado['fipezap_rec'].iloc[-1]
            variacao_simulada = ((predicao - valor_atual) / valor_atual) * 100
            
            # Mostrar resultados
            st.success("Simulação Concluída!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Valor Atual", f"{valor_atual:.2f}")
            
            with col2:
                st.metric("Valor Simulado", f"{predicao:.2f}")
            
            with col3:
                st.metric("Variação", f"{variacao_simulada:+.2f}%")
            
            # Interpretação do resultado
            if variacao_simulada > 0:
                st.success(f"📈 O cenário simulado indica uma valorização de {variacao_simulada:.2f}% nos preços imobiliários.")
            else:
                st.warning(f"📉 O cenário simulado indica uma desvalorização de {abs(variacao_simulada):.2f}% nos preços imobiliários.")
            
            # Gráfico comparativo
            fig_sim = go.Figure()
            
            fig_sim.add_bar(
                x=['Valor Atual', 'Cenário Simulado'],
                y=[valor_atual, predicao],
                marker_color=['blue', 'red' if variacao_simulada > 0 else 'orange']
            )
            
            fig_sim.update_layout(
                title="Comparação: Atual vs Cenário Simulado",
                yaxis_title="Índice FIPEZAP"
            )
            
            st.plotly_chart(fig_sim, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Dashboard desenvolvido para análise do mercado imobiliário de Recife | 📊 **Dados atualizados até**: " + 
           df['data'].max().strftime('%B/%Y'))