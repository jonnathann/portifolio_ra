#!/usr/bin/env python
# coding: utf-8

# ## Análise de Segmentação de Clientes

# #### Contexto dos Dados e Problema a Ser Resolvido:
# 
# O estudo foi realizado com dados de clientes de uma empresa, com o objetivo de identificar diferentes perfis de clientes e, a partir disso, segmentá-los para personalizar campanhas de marketing. Esses dados incluem informações sobre o comportamento de compra dos clientes, como a recência (quanto tempo desde a última compra), a frequência (quantas vezes o cliente comprou) e o valor monetário (quanto o cliente gastou). O problema que buscamos resolver é como otimizar as campanhas de marketing para cada perfil de cliente, a fim de melhorar a retenção, o engajamento e o aumento do valor de vida útil do cliente (LTV).
# 
# #### Técnicas Utilizadas:
# 
# Para resolver esse problema, utilizamos a análise RFM (Recência, Frequência e Valor Monetário), uma técnica amplamente utilizada para segmentação de clientes. Com ela, conseguimos criar três métricas de pontuação para cada cliente, com base em seu comportamento de compra. A segmentação dos clientes foi feita em diferentes grupos, permitindo a personalização das campanhas de marketing para cada tipo de perfil.
# 
# Com isso, conseguimos identificar cinco segmentos distintos de clientes e sugerir campanhas de marketing personalizadas para cada um, visando maximizar o impacto das ações promocionais e melhorar os resultados de vendas da empresa.

# In[2]:


import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

df = pd.read_excel(url)


# #### Análise exploratória (EDA)

# ##### Visualizando as primeiras linhas dos dados

# In[3]:


df.head()


# ##### Atributos do conjunto de dados

# In[4]:


df.columns


# ##### Tamanho do conjunto de dados

# In[5]:


df.shape


# ##### Verificando os tipos de dados das colunas

# In[6]:


df.info()


# ##### Verificar se existem dados faltantes

# In[7]:


df.isnull().sum()


# ##### Tratando os dados faltantes

# In[8]:


#Adicinando a frase 'sem descrição' nos campos onde a descrição está faltando
df['Description'] = df['Description'].fillna('Sem descrição')

#Como tem clientes sem ID, vamos removê-los para não atrapalhar a segmentação
df.dropna(subset=['CustomerID'], inplace=True)


# ##### Verificando novamente a existência de dados faltantes

# In[9]:


df.isnull().sum()


# #### Calculando métricas RFM:
# 
# - Recência (R): Tempo desde a última compra do cliente.
# - Frequência (F): Número total de compras feitas pelo cliente.
# - Valor Monetário (M): Total gasto por cada cliente.

# ##### Recência (R)

# In[10]:


# Definir a data de referência (última data no conjunto de dados)
latest_date = df['InvoiceDate'].max()

# Calcular a recência de cada cliente (em dias)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days  # Última compra de cada cliente
})
rfm.rename(columns={'InvoiceDate': 'Recency'}, inplace=True)


# In[11]:


rfm.head()


# ##### Frequência (F)

# In[12]:


# Calcular a frequência de compras de cada cliente
rfm['Frequency'] = df.groupby('CustomerID').size()


# In[13]:


rfm.head()


# ##### Valor Monetário (M)

# In[14]:


# Calcular o valor gasto por cada cliente (Quantity * UnitPrice)
df['TotalSpend'] = df['Quantity'] * df['UnitPrice']

# Calcular o valor monetário total gasto por cliente
rfm['Monetary'] = df.groupby('CustomerID')['TotalSpend'].sum()


# In[15]:


rfm.head()


# #### Segmentação de clientes

# In[18]:


import pandas as pd
import numpy as np

# Tratamento dos dados (geralmente aqui você já faz a limpeza e formatação)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Definindo a data de referência para calcular a Recência
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Calculando RFM
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recência: dias desde a última compra
    'InvoiceNo': 'count',  # Frequência: número de compras
    'TotalPrice': 'sum'  # Valor Monetário: total gasto
})

rfm.rename(columns={'InvoiceDate': 'Recencia', 'InvoiceNo': 'Frequencia', 'TotalPrice': 'Valor'}, inplace=True)

# Atribuindo Scores para Recência, Frequência e Valor Monetário com base em quantis (5 quantis)
rfm['Recencia_Score'] = pd.qcut(rfm['Recencia'], 5, labels=[5, 4, 3, 2, 1])
rfm['Frequencia_Score'] = pd.qcut(rfm['Frequencia'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
rfm['Valor_Score'] = pd.qcut(rfm['Valor'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Calculando o Score Total (Soma dos Scores de Recência, Frequência e Valor)
rfm['Total_Score'] = rfm['Recencia_Score'].astype(int) + rfm['Frequencia_Score'].astype(int) + rfm['Valor_Score'].astype(int)

# Segmentação com base no Total_Score
def segmentacao(row):
    if row['Total_Score'] >= 13:
        return 'Segmento 1: Clientes mais valiosos'
    elif row['Total_Score'] >= 9:
        return 'Segmento 2: Clientes potenciais'
    elif row['Total_Score'] >= 5:
        return 'Segmento 3: Clientes regulares'
    else:
        return 'Segmento 4: Clientes inativos'

rfm['Segmento'] = rfm.apply(segmentacao, axis=1)

#Salvando o arquivo na pasta processed
rfm.to_csv('../data/processed/rfm_segmentation.csv')

# Exibindo o resultado final
rfm.head()


# #### Resultados

# Com base na tabela RFM (Recência, Frequência e Valor Monetário), é possível aplicar várias campanhas de marketing segmentadas para melhorar o engajamento dos clientes. Aqui estão algumas sugestões de campanhas com base em cada segmento identificado:
# 
# ##### Segmento 1: Clientes mais valiosos
#     Campanhas:
#     
#     Fidelidade: Recompensas exclusivas.
#     
#     Ofertas personalizadas: Descontos em produtos premium.
#     
#     Agradecimento: Cartão ou bônus de compras.
#     
#     Acesso antecipado: Novos produtos ou promoções.
# 
# ##### Segmento 2: Clientes potenciais
#     Campanhas:
#     
#     Reengajamento: Cupons ou promoções para compras mais frequentes.
#     
#     Promoções de frequência: Descontos progressivos.
#     
#     Novidades: E-mail marketing com novos produtos.
#     
#     Recomendação: Benefícios por indicar amigos.
# 
# ##### Segmento 3: Clientes regulares
#     Campanhas:
#     
#     Incentivo a mais compras: Promoções para aumentar a frequência.
#     
#     Valor agregado: Descontos em compras acima de valor específico.
#     
#     Cross-selling: Ofereça produtos complementares.
# 
# ##### Segmento 4: Clientes inativos
#     Campanhas:
#     
#     Reativação: Descontos ou ofertas exclusivas.
#     
#     Recuperação de carrinho: Lembretes para completar compras.
#     
#     Novos produtos: E-mail com sugestões baseadas em compras anteriores.
# 
# ##### Segmento 5: Clientes de baixo valor
#     Campanhas:
#     
#     Incentivo à primeira compra: Descontos para aumentar valor.
#     
#     Despertar interesse: Oferta introdutória ou desconto no primeiro pedido.
#     
#     Lembrete de produtos: Reforçar itens no carrinho ou de interesse.

# #### Conclusão
# 
# A análise de segmentação de clientes utilizando a metodologia RFM (Recência, Frequência e Valor Monetário) revelou padrões distintos de comportamento entre os clientes. Com base nesses segmentos, foram sugeridas campanhas de marketing personalizadas para aumentar o engajamento e a fidelização dos clientes. As estratégias incluem desde programas de fidelidade para clientes mais valiosos, até campanhas de reativação para clientes inativos. A implementação dessas campanhas tem o potencial de melhorar as vendas e otimizar os recursos de marketing, criando uma abordagem mais eficaz e direcionada para cada grupo de clientes.
