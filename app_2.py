import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Carregando o DataFrame longo para os gráficos de insights
df_longo = pd.read_csv("df_longo_app.csv", delimiter=';')

# 1. Configurações Iniciais e Carregamento de Recursos
st.set_page_config(page_title="Hospital Predictor - Diagnóstico de Obesidade", 
                   layout="wide")

@st.cache_resource
def load_resources():
    # Carrega o modelo, o scaler e o dicionário de mapeamento
    model = joblib.load("modelo_xgboost.pkl")
    scaler = joblib.load("scaler.pkl")
    df_dict = pd.read_csv("dicionario.csv", delimiter=';')
    return model, scaler, df_dict

model, scaler, df_dict = load_resources()

# Função auxiliar para filtrar opções do dicionário para os Selectboxes
def obter_opcoes(coluna):
    subset = df_dict[df_dict['cd_variavel'] == coluna]
    # Retorna dicionário {Descrição para o usuário: Código para o modelo}
    return dict(zip(subset['ds_categoria'], subset['sk_categoria']))

# --- GRÁFICOS DE INSIGHTS ---
class GraficosInsights:
    def __init__(self, df_longo):
        self.df_longo = df_longo
        self.tabela = None
        self.titulo = None
        self.fig = None
        self.totais_categoria = None

    def gerar_graficos(self, variavel):
        self.tabela = self.df_longo[self.df_longo['cd_variavel'] == variavel]
        # Pega o nome descritivo da variável (ds_variavel) se existir, senão usa o nome da variável
        if not self.tabela.empty and 'ds_variavel' in self.tabela.columns:
            self.titulo = self.tabela['ds_variavel'].iloc[0]
        else:
            self.titulo = variavel.replace("_", " ").capitalize()
        if not self.tabela.empty:
            # Calcula a tabela de percentuais por categoria e grau de obesidade
            tabela_pct = self.tabela.groupby(['ds_categoria', 'ds_obesidade']).size().reset_index(name='count')
            total_por_categoria = tabela_pct.groupby('ds_categoria')['count'].transform('sum')
            tabela_pct['percentual'] = 100 * tabela_pct['count'] / total_por_categoria
            # Para exibir total por coluna (categoria)
            totais = tabela_pct.groupby('ds_categoria').agg({'count': 'sum'}).reset_index()
            total_geral = totais['count'].sum()
            totais['percentual_total'] = 100 * totais['count'] / total_geral
            # Tabela de totais por categoria (ex: quantidade de pessoas por sexo)
            self.totais_categoria = totais.rename(columns={'ds_categoria': self.titulo, 'count': 'Quantidade', 'percentual_total': '% do total'})
            self.totais_categoria['% do total'] = self.totais_categoria['% do total'].map(lambda x: f"{x:.1f}")
            # Gráfico de barras empilhadas (stacked bar) com Plotly, mostrando percentual e count no hover
            self.fig = px.bar(
                tabela_pct,
                x='ds_categoria',
                y='percentual',
                color='ds_obesidade',
                title=f"Distribuição percentual dos graus de obesidade por {self.titulo}",
                labels={'ds_categoria': self.titulo, 
                        'percentual': 'Percentual (%)', 
                        'ds_obesidade': 'Grau de obesidade'},
                text_auto='.1f',
                custom_data=['count']
            )
            self.fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Grau: %{legendgroup}<br>Percentual: '
                '%{y:.1f}%<br>Contagem: %{customdata[0]}<extra></extra>'
            )
            self.fig.update_layout(barmode='stack', yaxis=dict(range=[0, 100]))
            # Adiciona anotação de total e percentual total acima de cada coluna
            for i, row in totais.iterrows():
                self.fig.add_annotation(
                    x=row['ds_categoria'],
                    y=102,
                    text=f"Qtd: {int(row['count'])}<br>{row['percentual_total']:.1f}% do total",
                    showarrow=False,
                    font=dict(size=11, color="black"),
                    align="center"
                )
        else:
            self.tabela = None
            self.fig = None

# --- INTERFACE ---
st.title("🩺 Sistema Inteligente de Apoio ao Diagnóstico")
st.markdown("---")

# Abas para separar a ferramenta de predição dos insights visuais
tab_pred, tab_dash = st.tabs(["Previsão Individual", "Visão Analítica (Negócio)"])

with tab_pred:
    st.subheader("Entrada de Dados do Paciente")
    
    with st.form("diagnostico_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Dados Biométricos**")
            # Entradas numéricas diretas
            idade = st.number_input("Idade (anos)", 13, 60, 25)
            altura = st.number_input("Altura (cm)", 100, 220, 175)
            peso = st.number_input("Peso atual (kg)", 30, 300, 80)
            
            # Selectbox dinâmico usando o dicionário
            map_genero = obter_opcoes("gender")
            genero = st.selectbox("Sexo biológico", options=list(map_genero.keys()))

        with col2:
            st.write("**Histórico e Alimentação**")
            map_familia = obter_opcoes("family_history")
            historico_fam = st.selectbox("Histórico familiar de excesso de peso?", 
                                         options=list(map_familia.keys()))
            
            map_favc = obter_opcoes("favc")
            favc = st.selectbox("Consumo de alimentos com alto teor calórico", 
                                options=list(map_favc.keys()))

            map_vegetais = obter_opcoes("fcvc")
            vegetais = st.selectbox("Frequência de consumo de vegetais", 
                                    options=list(map_vegetais.keys()))
            
            map_refeicoes = obter_opcoes("ncp")
            refeicoes = st.selectbox("Número de refeições principais por dia", 
                                     options=list(map_refeicoes.keys()))

            map_caec = obter_opcoes("caec")
            caec = st.selectbox("Consumo de alimentos entre as refeições", 
                                options=list(map_caec.keys()))   

            map_scc = obter_opcoes("scc")
            scc = st.selectbox("Monitora a ingestão de calorias?", 
                               options=list(map_scc.keys())) 

        with col3:
            st.write("**Estilo de Vida e Hábitos**")
            map_transporte = obter_opcoes("mtrans")
            transporte = st.selectbox("Meio de transporte habitual", 
                                      options=list(map_transporte.keys()))
            
            map_atv_fisica = obter_opcoes("faf")
            atv_fisica = st.select_slider("Frequência semanal de atividade física", 
                                          options=list(map_atv_fisica.keys()))
            
            map_alcool = obter_opcoes("calc")
            alcool = st.selectbox("Consumo de bebida alcoólica", 
                                  options=list(map_alcool.keys()))

            map_tue = obter_opcoes("tue")
            tue = st.selectbox("Tempo gasto em atividades sedentárias (TV, computador, celular)", 
                               options=list(map_tue.keys()))

            # Não será consumido pelo modelo, mas é um dado interessante para o médico
            map_smoke = obter_opcoes("smoke")
            smoke = st.selectbox("Fumante?", options=list(map_smoke.keys()))

            map_ch2o = obter_opcoes("ch2o")
            ch2o = st.selectbox("Consumo diário de água", options=list(map_ch2o.keys()))

        enviar = st.form_submit_button("Gerar Predição")

    if enviar:
        # --- PROCESSAMENTO DOS DADOS (REPLICANDO O NOTEBOOK) ---
        
        # 1. Cálculo do IMC
        imc_calculado = peso / ((altura / 100) ** 2)
        
        # 2. Engenharia de Variável: Transporte Ativo
        # Atribuímos 1 para Bike/Walking e 0 para os demais, conforme sua lógica
        transporte_ativo = 1 if transporte in ["Bicicleta", "A pé"] else 0

        # Convertemos o "Sempre" em "Frequentemente" para o modelo, 
        # já que ele foi treinado sem essa categoria
        if atv_fisica == "Sempre":
            atv_fisica = "Frequentemente"
        
        # 3. Montagem do vetor de entrada com os códigos do dicionário,
        # retornando sk_categoria para cada variável categórica e mantendo as numéricas como estão
        dados_entrada = {
            "gender": map_genero[genero],
            "age": idade,
            "height": altura,
            "family_history": map_familia[historico_fam],
            "favc": map_favc[favc],
            "fcvc": map_vegetais[vegetais],
            "ncp": map_refeicoes[refeicoes],
            "caec": map_caec[caec], 
            "ch2o": map_ch2o[ch2o], 
            "scc": map_scc[scc], 
            "faf": map_atv_fisica[atv_fisica],
            "tue": map_tue[tue], 
            "calc": map_alcool[alcool],
            "imc": imc_calculado,
            "mtrans_ativo": transporte_ativo
        }

        # 4. Ajuste de Escala (Normalização)
        df_final = pd.DataFrame([dados_entrada])
        # Reordenação para bater com as colunas do modelo
        colunas_modelo = ["gender", "age", "height", "family_history", "favc", "fcvc", "ncp", 
                         "caec", "ch2o", "scc", "faf", "tue", "calc", "mtrans_ativo", "imc"]
        df_final = df_final[colunas_modelo]
        
        # Aplica o Scaler nas numéricas
        df_final[["age", "height", "imc"]] = scaler.transform(df_final[["age", "height", "imc"]])

        # 5. Predição Final
        pred_idx = model.predict(df_final)[0]
        # Como no XGBoost você usou labels 0-6, somamos 1 para bater com o sk_categoria (1-7)
        map_obesidade = df_dict[df_dict['cd_variavel'] == 'obesity']
        resultado_final = map_obesidade[map_obesidade['sk_categoria'] == (pred_idx + 1)]['ds_categoria'].values[0]

        # Exibição
        st.success(f"### Diagnóstico Sugerido: **{resultado_final}**")
        st.info(f"O IMC calculado para este paciente é de **{imc_calculado:.2f}**.")    

with tab_dash:
    st.subheader("Insights Médicos e Panorama da Saúde")
    st.markdown("Resumo dos principais gatilhos comportamentais identificados no estudo.")
    
# exibindo os gráficos de insights
    graficos = GraficosInsights(df_longo)
    variaveis_para_graficos = ["gender", "family_history", "favc", "fcvc", 
                               "ncp", "caec", "ch2o", "scc", "smoke", "faf", "tue", 
                               "calc", "mtrans"]
    for var in variaveis_para_graficos:
        graficos.gerar_graficos(var)
        if graficos.fig is not None:
            st.plotly_chart(graficos.fig, use_container_width=True)
        if graficos.totais_categoria is not None:
            st.dataframe(graficos.totais_categoria, use_container_width=True, hide_index=True)



