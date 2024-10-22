# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 00:26:22 2024

@author: Paulo
"""

#%% Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
import matplotlib as plt1
import plotly.graph_objects as go
pio.renderers.default='browser'

#%% Importando o banco de dados

# Objetivo: agrupar os clientes de uma operadora de cartão de crédito

dados_base_numeros = pd.read_excel('base_numeros_final.xlsx')
dados_base_clientes = pd.read_excel('base_clientes.xlsx')
## Fonte: https://www.kaggle.com/datasets/aryashah2k/cr'edit-card-customer-data

#%% Visualizando informações sobre os dados e variáveis

# Estrutura do banco de dados

#print(dados_base_numeros.info())
print(base_numeros_final.info())


#%% Estatísticas descritivas das variáveis

# Primeiramente, vamos excluir as variáveis que não serão utilizadas

#dados_transações_cluster = dados_base_numeros.drop(columns=['id_tran','cpf'])
dados_transações_cluster = base_numeros_final.drop(columns= ['CPF', 'Faixa Etaria'])

# Obtendo as estatísticas descritivas das variáveis

tab_descritivas = dados_transações_cluster.describe().T
# Vamos padronizar as variáveis antes da clusterização!

#%% Padronização por meio do Z-Score

# Aplicando o procedimento de ZScore
transações_pad = dados_transações_cluster.apply(zscore, ddof=1)

# Visualizando o resultado do procedimento na média e desvio padrão
print(round(transações_pad.mean(), 3))
print(round(transações_pad.std(), 3))


# Poderíamos fazer vários boxplot em um mesmo gráfico

var_boxplot = transações_pad[['VALOR_TOTAL_GASTO', 'IDADE', 'RENDA_MENSAL']]

sns.boxplot(data=var_boxplot, width = 0.6, palette='rocket')
plt.xlabel('Produtos',fontsize=12)
plt.ylabel('Valores',fontsize=12)
plt.show()

# Vamos torná-lo mais informativo

fig = px.box(var_boxplot, 
             width = 900)
fig.update_layout(title='BOXPLOT',
                  xaxis_title='Produtos',
                  yaxis_title='Valores',
                  plot_bgcolor='lightblue')
fig.show()



#%% Gráfico de calor

# Vamos gerar um gráfico de calor que distingue informações por meio de cores
# O banco de dados contém informações sobre a quantidade vendida em 3 produtos

#vendas_regional = pd.read_excel("(2) vendas_regiao.xlsx")

# Inicialmente, vamos selecionar as variáveis quantitativas do banco de dados

#vendas_reg = vendas_regional[['produtoA','produtoB','produtoC']]

# Vamos gerar o gráfico de calor no contexto das correlações entre variáveis
# Portanto, primeiramente, vamos criar a matriz de correlações de Pearson
# Lembrando: selecionar apenas as variáveis quantitativas da base de dados

corr = transações_pad.corr()

# Vamos elaborar um gráfico de calor (heatmap) com o plotly

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = corr.columns,
        y = corr.index,
        z = np.array(corr),
        text=corr.values,
        texttemplate='%{text:.2f}',
        colorscale='ice'))

fig.update_layout(
    height = 600,
    width = 600)

fig.show()

#%% Gráfico 3D das observações

fig = px.scatter_3d(transações_pad, 
                    x='VALOR_TOTAL_GASTO', 
                    y='IDADE', 
                    z='RENDA_MENSAL')
fig.show()
#%% Gráfico 3D das observações

fig = plt.bar()
fig.show()

#%% Gráfico 3D das observações

fig = px.scatter(transações_pad,
                     x='Idade')
fig.show()

#%% Gráfico 3D das observações

fig = px.scatter(transações_pad,
                     x='Renda_CPF')
fig.show()

#%% Identificação da quantidade de clusters (Método Elbow)

elbow = []
K = range(1,50) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(transações_pad)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,50)) # ajustar range
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

#%% Identificação da quantidade de clusters (Método da Silhueta)

silhueta = []
I = range(2,50) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=20).fit(transações_pad)
    silhueta.append(silhouette_score(transações_pad, kmeansSil.labels_))

plt.figure(figsize=(16,8))
plt.plot(range(2, 50), silhueta, color = 'purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()

#%% Cluster Não Hierárquico K-means

# Vamos considerar 5 clusters, considerando as evidências anteriores!

kmeans_final = KMeans(n_clusters = 5, init = 'random', random_state=100).fit(transações_pad)

# Gerando a variável para identificarmos os clusters gerados

kmeans_clusters = kmeans_final.labels_
dados_transações_cluster['cluster_kmeans'] = kmeans_clusters
transações_pad['cluster_kmeans'] = kmeans_clusters
dados_transações_cluster['cluster_kmeans'] = dados_transações_cluster['cluster_kmeans'].astype('category')
transações_pad['cluster_kmeans'] = transações_pad['cluster_kmeans'].astype('category')

#%% Análise de variância de um fator (ANOVA)

# Interpretação do output:

## cluster_kmeans MS: indica a variabilidade entre grupos
## Within MS: indica a variabilidade dentro dos grupos
## F: estatística de teste (cluster_kmeans MS / Within MS)
## p-unc: p-valor da estatística F
## se p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais

# Avg_Credit_Limit
pg.anova(dv='valor', 
         between='cluster_kmeans', 
         data=transações_pad,
         detailed=True).T

# Total_Credit_Cards
pg.anova(dv='Idade', 
         between='cluster_kmeans', 
         data=transações_pad,
         detailed=True).T

# Total_visits_bank
pg.anova(dv='Renda_CPF', 
         between='cluster_kmeans', 
         data=transações_pad,
         detailed=True).T

#%% Gráfico 3D dos clusters

# Perspectiva 1

fig = px.scatter_3d(dados_transações_cluster, 
                    x='valor', 
                    y='Idade', 
                    z='Renda_CPF',
                    color='cluster_kmeans')
fig.show()

#%% Cluster Hierárquico Aglomerativo: single linkage + distância cityblock

# Gerando o dendrograma

plt.figure(figsize=(16,8))
dend_sing = sch.linkage(transações_pad, method = 'single', metric = 'cityblock')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 60, labels = list(transações_pad))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Valor', fontsize=16)
plt.ylabel('Renda_CPF', fontsize=16)
plt.axhline(y = 60, color = 'red', linestyle = '--')
plt.show()

# Opções para o método de encadeamento ("method"):
    ## single
    ## complete
    ## average

# Opções para as distâncias ("metric"):
    ## euclidean
    ## sqeuclidean
    ## cityblock
    ## chebyshev
    ## canberra
    ## correlation

# Criando a variável que indica os clusters no banco de dados

cluster_sing = AgglomerativeClustering(n_clusters = 3, metric = 'cityblock', linkage = 'single')
indica_cluster_sing = cluster_sing.fit_predict(varejo)
transações_pad['cluster_single'] = indica_cluster_sing
transações_pad['cluster_single'] = transações_pad['cluster_single'].astype('category')

#Análise Fatorial (Teste Esferificidade de Bartlett)

#%% Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install factor_analyzer
!pip install sympy
!pip install scipy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install pingouin
!pip install pyshp

#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go

#%% Importando o banco de dados

base_clientes = pd.read_excel("base_clientes.xlsx")
base_transacoes_tratada = pd.read_excel("base_transacoes_tratada.xlsx")
base_join = pd.read_excel("base_join.xlsx")
base_numeros = pd.read_excel("base_numeros.xlsx")
base_clientes_tratada = pd.read_excel("base_clientes_tratada.xlsx")
base_numeros_final = pd.read_excel("base_numeros_final.xlsx")


#%%Group by na tabela de clientes

#tab_base_clientes = base_clientes.groupby
#print('tab_base_clientes')

#base_clientes_pd = pd.

#%% Estatísticas descritiva das variáveis

#tab_desc_clientes = tab_base_clientes.describe()
tab_desc_transacoes_tratada = base_transacoes_tratada.describe()
tab_desc_base_join = base_join.describe()
tab_desc_clientes_tratada = base_clientes_tratada.describe()
tab_desc_numeros_final = base_numeros_final.describe()
tab_desc_numeros = base_numeros.describe()

#%% Analisando as correlações de Pearson

# Matriz de correlações

corr = base_numeros.corr()

# Gráfico interativo

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = corr.columns,
        y = corr.index,
        z = np.array(corr),
        text=corr.values,
        texttemplate='%{text:.3f}',
        colorscale='viridis'))

fig.update_layout(
    height = 750,
    width = 750,
    yaxis=dict(autorange="reversed"))

fig.show()

#%% Selecionando as variáveis de interesse para a análise

# Vamos deixar o preço das casas de fora da análise fatorial! 

base_numeros_pca = base_numeros.drop(columns=['id_tran','cpf'])


#%% Teste de Esfericidade de Bartlett

bartlett, p_value = calculate_bartlett_sphericity(base_numeros_pca)

print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')
