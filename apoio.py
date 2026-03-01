import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Apoio:

    def __init__(self):
        self.tabela = None
        self.ordem = None
        self.titulo = None
        self.plt = plt
        self.pd = pd
        self.np = np
        self.sns = sns
        self.axes = None
        self.df_longo = None
        self.ordem_grau_obesidade = None
    
    def arredonda_inteiro_mais_proximo(self, x):
        return self.np.floor(x + 0.5)

    def gerar_df_longo(self, df_pesquisa, df_dicionario):
        # transforma em um dataset longo
        id_vars = ["age", "height", "weight", "obesity"]

        self.df_longo = df_pesquisa.melt(
            id_vars=id_vars,
            var_name="cd_variavel",
            value_name="nr_categoria"
        )

        # filtra o dicionário para obter apenas dados os rótulos de obesidade
        df_obesidade = df_dicionario.loc[df_dicionario['cd_variavel'] == 'obesity']

        mapa_descricao_obesidade = (
            df_obesidade
            .set_index("nr_categoria")["ds_categoria"]
        )

        self.ordem_grau_obesidade = (
            df_obesidade
            .sort_values("sk_categoria")["ds_categoria"]
            .tolist()
        )

        self.df_longo["ds_obesidade"] = self.df_longo["obesity"].map(mapa_descricao_obesidade)

        # cruza dos dados com o dicionário
        self.df_longo = self.df_longo.merge(
            df_dicionario,
            on=["cd_variavel", "nr_categoria"],
            how="left"
        )

        self.gerar_df_longo_app()
    
    def filtrar_variavel(self, variavel):
        
        filtro = self.df_longo["cd_variavel"] == variavel

        df = (
            self.pd.crosstab(
                self.df_longo.loc[filtro, "ds_categoria"],
                self.df_longo.loc[filtro, "ds_obesidade"],
                normalize="index"
            )
            .mul(100)
            .reindex(columns=self.ordem_grau_obesidade)
        )

        return df

    def grafico_tipos_obesidade_por_sexo(self):

        tabela = self.filtrar_variavel('gender')
        
        cores = {
            "Masculino": "#6BAED6",
            "Feminino": "#F4A6C1"
        }

        ax = tabela.T.plot(
            kind="bar",
            figsize=(14,4),
            color=[cores[col] for col in tabela.T.columns])
        
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.1f%%",
                label_type="edge",
                padding=3,
                fontsize=9,        
            )

        ax.set_ylabel("Percentual (%)")
        ax.set_xlabel("Tipos de Obesidade")        
        ax.legend(
            title="Sexo",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False
        )
        ax.margins(y=0.1)

        self.plt.xticks(rotation=0)
        self.plt.title("Distribuição percentual dos tipos de obesidade por sexo")
        self.plt.tight_layout()
        self.plt.show()

    def grafico_frequencia_idade_e_faixa_etaria(self):

        tabela = self.df_longo.loc[self.df_longo.cd_variavel == "gender"].copy()
        coluna ='age'

        # Criando faixas etárias de 10 em 10 anos
        bins = self.np.arange(0, tabela[coluna].max() + 10, 10)
        labels = [f"{i}-{i+9}" for i in bins[:-1]]

        tabela["faixa_etaria"] = self.pd.cut(
            tabela[coluna],
            bins=bins,
            labels=labels,
            right=False
        )

        # Criando figura com dois gráficos lado a lado
        fig, axes = self.plt.subplots(1, 2, figsize=(14, 4))

        # histograma
        self.sns.histplot(
            data=tabela,
            x=coluna,
            bins=50,
            ax=axes[0]
        )

        axes[0].set_title("Frequencia de Respondentes da pesquisa por Idade")
        axes[0].set_xlabel("Idade")
        axes[0].set_ylabel("Quantidade")

        # gráfico por faixa etária
        self.sns.countplot(
            data=tabela,
            x="faixa_etaria",
            ax=axes[1]
        )

        axes[1].set_title("Distribuição por Faixa Etária (10 em 10 anos)")
        axes[1].set_xlabel("Faixa Etária")
        axes[1].set_ylabel("Quantidade")
        axes[1].tick_params(axis="x", rotation=0)

        # adicionando os rótulos
        for container in axes[1].containers:
            axes[1].bar_label(
                container,
                fmt="%.0f",
                fontsize=9,                
            )

        self.plt.tight_layout()
        self.plt.show()

    def grafico_tipos_obesidade_por_faixa_etaria(self):
        
        self.tabela = self.df_longo.loc[self.df_longo.cd_variavel == "gender"].copy()
        
        bins = self.np.arange(0, self.tabela["age"].max() + 10, 10)
        labels = [f"{i}-{i+9}" for i in bins[:-1]]

        self.tabela["faixa_etaria"] = self.pd.cut(
            self.tabela["age"],
            bins=bins,
            labels=labels,
            right=False
        )

        df_age = self.pd.crosstab(
                self.tabela["faixa_etaria"],
                self.tabela["ds_obesidade"],
                normalize="index"
            ).mul(100).round(1)

        self.tabela = df_age.reindex(columns=self.ordem_grau_obesidade)
        self.ordem = df_age.index
        self.titulo = "Faixa Etária" 

        self.gerar_graficos()

    def gerar_graficos(self):
        
        self.tabela = self.tabela.loc[self.ordem]

        fig, axes = self.plt.subplots(1, 2, figsize=(15,5), sharey=True)

        self.ax1 = axes[0]
        self.ax2 = axes[1]

        self.graficoA()
        self.graficoB()

        self.plt.tight_layout()
        self.plt.show()

    def graficoA(self):

        # Barras empilhadas
        self.tabela.plot(
            kind="bar",
            stacked=True,
            ax=self.ax1
        )

        self.ax1.set_title(f"Distribuição percentual dos tipos de obesidade\n por {self.titulo}")
        self.ax1.set_xlabel(self.titulo)
        self.ax1.set_ylabel("Percentual (%)")
        self.ax1.set_ylim(0, 100)

        for container in self.ax1.containers:
            labels = [
                f"{v:.1f}%" if v >= 5 else ""
                for v in container.datavalues
            ]
            self.ax1.bar_label(
                container,
                labels=labels,
                label_type="center",
                fontsize=8,
                clip_on=False
            )

        self.ax1.legend(
            title="tipos de obesidade",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.35),
            ncol=4,
            frameon=False
        )

        self.ax1.tick_params(axis="x", labelrotation=0)

    def graficoB(self):

        grupos = {
            "Baixo / Normal": ["Abaixo do peso", "Peso normal"],
            "Sobrepeso": ["Sobrepeso nível I", "Sobrepeso nível II"],
            "Obesidade": ["Obesidade tipo I", "Obesidade tipo II", "Obesidade tipo III"]
        }

        tabela_agrupada = self.pd.DataFrame({
            grupo: self.tabela[cols].sum(axis=1)
            for grupo, cols in grupos.items()
        })

        # Barras agrupadas
        tabela_agrupada.plot(
            kind="bar",
            ax=self.ax2,
            width=0.8
        )

        self.ax2.set_title(f"Agrupamento dos tipos de obesidade\n por {self.titulo}")
        self.ax2.set_xlabel(self.titulo)
        self.ax2.set_ylabel("Percentual (%)")
        self.ax2.set_ylim(0, 100)

        for container in self.ax2.containers:
            labels = [
                f"{v:.1f}%" for v in container.datavalues
            ]
            self.ax2.bar_label(container, labels=labels, fontsize=8)

        self.ax2.legend(
            title="tipos de obesidade",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.35),
            ncol=4,
            frameon=False
        )

        self.ax2.tick_params(axis="x", labelrotation=0)

    def grafico_historico_familiar(self):

        tabela = self.filtrar_variavel("family_history")

        ax = tabela.T.plot(kind="bar", figsize=(14,4))

        self.plt.title("Distribuição percentual dos tipos de obesidade segundo histórico familiar")
        self.plt.ylabel("Percentual")
        self.plt.xlabel("tipos de obesidade")
        self.plt.legend(title="Histórico Familiar")
        self.plt.xticks(rotation=0)

        # adicionando os rótulos
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.1f%%",
                fontsize=9
            )

        self.plt.show()

    def grafigo_alimentos_altamente_caloricos(self):
        self.tabela = self.filtrar_variavel("favc")
        self.ordem = ["Sim", "Não"]
        self.titulo = "consumo frequente de alimentos altamente calóricos"
        self.gerar_graficos()

    def grafico_consumo_de_vegetais(self):
        self.tabela = self.filtrar_variavel("fcvc")
        self.ordem = ["Raramente", "Às vezes", "Sempre"]
        self.titulo = "frequência de consumo de vegetais"
        self.gerar_graficos()

    def grafico_refeicoes_principais(self):
        self.tabela = self.filtrar_variavel("ncp")
        self.ordem = ["Uma refeição","Duas refeições","Três refeições","Quatro ou mais refeições"]
        self.titulo = "número de refeições principais por dia"
        self.gerar_graficos()

    def grafico_alimento_entre_refeicoes(self):
        self.tabela = self.filtrar_variavel("caec")
        self.ordem = ["Não consome","Às vezes","Frequentemente","Sempre"]
        self.titulo = "consumo de alimento entre as refeições"
        self.gerar_graficos()

    def grafico_pratica_atividade_fisica(self):
        self.tabela = self.filtrar_variavel("faf")
        self.ordem = ["Nenhuma","1–2 vezes","3–4 vezes","5 vezes ou mais"]
        self.titulo = "frequência semanal de atividade física"
        self.gerar_graficos()

    def grafico_tempo_uso_eletronico(self):
        self.tabela = self.filtrar_variavel("tue")
        self.ordem = ["0–2 horas/dia","3–5 horas/dia","Mais de 5 horas/dia",]
        self.titulo = "tempo diário usando dispositivos eletrônicos"
        self.gerar_graficos()

    def grafico_meio_transporte_habitual(self):
        self.tabela = self.filtrar_variavel("mtrans")
        self.ordem = ["Carro","Moto","Bicicleta","Transporte público","A pé"]
        self.titulo = "meio de transporte habitual"
        self.gerar_graficos()

    def grafico_consumo_de_agua(self):
        self.tabela = self.filtrar_variavel("ch2o")
        self.ordem = ["Menos de 1 litro/dia", "Entre 1 e 2 litros/dia", "Mais de 2 litros/dia"]
        self.titulo = "consumo diário de água"
        self.gerar_graficos()

    def grafico_consumo_bebida_alcoolica(self):
        self.tabela = self.filtrar_variavel("calc")
        self.ordem = ["Não bebe","Às vezes","Frequentemente","Sempre"]
        self.titulo = "consumo de bebida alcoólica"
        self.gerar_graficos()

    def grafico_habito_de_fumar(self):
        self.tabela = self.filtrar_variavel("smoke")
        self.ordem = ["Fuma", "Não fuma"]
        self.titulo = "hábito de fumar"
        self.gerar_graficos()

    def grafico_monitoramento_ingestao_calorica(self):
        self.tabela = self.filtrar_variavel("scc")
        self.ordem = ["Sim", "Não"]
        self.titulo = "monitora a ingestão calórica diária"
        self.gerar_graficos()

    def gerar_df_longo_app(self):
        df_longo = self.df_longo.copy()

        df_longo.to_csv('df_longo_app.csv', sep=';', index=False)