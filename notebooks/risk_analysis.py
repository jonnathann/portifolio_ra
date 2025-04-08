#!/usr/bin/env python
# coding: utf-8

# ## Detecção de Fraude em Dados Financeiros: Aplicação de Oversampling, Classificação e K-Means para Identificação de Padrões

# ### Análise Exporatória dos dados

# #### Contexto
# Empresas de cartão de crédito precisam identificar transações fraudulentas para evitar cobranças indevidas.
# 
# #### Conteúdo
# O conjunto de dados contém transações de cartões de crédito realizadas por portadores europeus em setembro de 2013. Ele inclui 492 fraudes de um total de 284.807 transações, sendo que as fraudes representam 0,172% do total. As variáveis de entrada são numéricas e resultam de uma transformação PCA, com exceção dos recursos 'Time' (tempo em segundos desde a primeira transação) e 'Amount' (valor da transação). A variável 'Class' indica se a transação é fraudulenta (1) ou não (0). Devido ao desequilíbrio de classes, recomenda-se usar a Área Sob a Curva de Precisão-Recall (AUPRC) para avaliar a precisão, em vez da matriz de confusão.
# 
# ##### link para o conjunto de dados: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# #### Importando dados do Kaggle
# Nessa parte, estamos fazendo o download do conjunto de dados para a realização das análises. O arquivo que contém o conjunto de dados se chama creditcard.csv. Também o carregamos em um dataframe para realizar uma análise exploratória mais aprofundada.

# In[1]:


import kagglehub as kag 
import pandas as pd


# In[2]:


path = kag.dataset_download("mlg-ulb/creditcardfraud")


# In[3]:


file_csv = f"{path}/creditcard.csv"


# #### Quais variáveis e sua quantidade?
# Neste ponto, verificamos que o conjunto total possui 31 variáveis, sendo que três delas são conhecidas: Time (tempo em segundos desde a primeira transação), Amount (valor da transação) e Class (classe da observação, sendo [0] para não fraude e [1] para fraude). As variáveis de V1 até V28, conforme descrito no enunciado dos dados, são componentes de um PCA que foi aplicado aos dados antes de serem disponibilizados.

# In[4]:


df = pd.read_csv(file_csv)


# In[5]:


df.columns


# In[6]:


df.columns.shape[0]


# #### Verificando o desbalanceamento dos dados
# Ao verificarmos o nível de desbalanceamento, percebemos que existem 284.315 observações que representam transações do tipo "não fraude" e 492 transações do tipo "fraude". Em termos matemáticos, temos uma diferença de quantidade entre as classes de 283.823 para a classe "não fraude". Resumindo, temos um conjunto de dados muito desbalanceado, onde a classe "fraude" não é representativa em relação à classe "não fraude".

# In[7]:


df_count_class = pd.DataFrame(df.Class.value_counts().reset_index())
df_count_class.columns = ['Classes', 'Frequência']
df_count_class


# #### Verificando a distribuição das classes "fraude" e "não fraude"
# Visualizando as distribuições das classes "fraude" e "não fraude".
# 
# 

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, width=0.2)
plt.title('Distribuição das Transações (Fraude vs Não Fraude)')
plt.xlabel('Classe (0: não fraude, 1: fraude)')
plt.ylabel('Quantidade de Transações')
plt.yscale('log')
plt.show()


# #### Verificando a proporção entre as classes "fraude" e "não fraude"
# Ao verificar a proporção de fraudes, vemos que o desbalanceamento é muito significativo, com aproximadamente 0,173%.

# In[10]:


fraud_ratio = df[df.Class == 1].shape[0]/df[df.Class == 0].shape[0]
print(f'Proporção de fraudes: {fraud_ratio:.5f}')


# #### Verificar a distribuição do valor das transações ('Amount') para fraudes e não fraudes
# Ao interpretar os boxplots, observamos que a coluna 'Amount' tem uma distribuição com simetria para cada classe, visto que as medianas (segundo quartil) estão mais localizadas no centro das caixas. Observamos também valores discrepantes (outliers) acima do limite superior de detecção para ambas as classes.

# In[11]:


plt.figure(figsize=(6, 4))
sns.boxplot(x='Class', y='Amount', data=df, width=0.2)
plt.title('Distribuição do valor das transações para fraudes e não fraudes')
plt.xlabel('Classe (0: Não Fraude, 1: Fraude)')
plt.ylabel('Valor da transação')
plt.yscale('log')
plt.show()


# #### Identificando quantas observações na coluna ('Amount') de cada uma das classes fraudes e não fraudes são outliers

# In[12]:


# Calculando os quartis e o IQR para a coluna 'Amount'
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1

# Definindo os limites superior e inferior para identificar outliers
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Identificando outliers para as fraudes
outliers_fraude = df[(df['Class'] == 1) & ((df['Amount'] < lower_limit) | (df['Amount'] > upper_limit))]

# Identificando outliers para as transações não fraudulentas
outliers_nao_fraude = df[(df['Class'] == 0) & ((df['Amount'] < lower_limit) | (df['Amount'] > upper_limit))]

# Exibindo os outliers encontrados
print(f"Outliers nas transações fraudulentas: {outliers_fraude.shape[0]}")
print(f"Outliers nas transações não fraudulentas: {outliers_nao_fraude.shape[0]}")


# #### Distribuição do valor das transações nas classes fraudes e não fraudes em transações atípicas (outliers)
# Ao analisar a distribuição abaixo, percebemos que, nas transações atípicas fraudulentas, os fraudadores sempre realizam transações com valores mais baixos, no intervalo de 0 a 2500 unidades. Nas transações legítimas, a distribuição é mais variada, com transações atípicas tanto altas quanto baixas. Um insight que podemos tirar é que os fraudadores, para não chamar a atenção, optam por realizar transações de valores baixos, seja de forma repetida ou não, de maneira frequente.

# In[13]:


plt.figure(figsize=(6, 4))
sns.histplot(df[df['Class'] == 0]['Amount'], bins=10, color='blue', alpha=0.6, label='Não Fraude', kde=True)
sns.histplot(df[df['Class'] == 1]['Amount'], bins=10, color='red', alpha=0.6, label='Fraude', kde=True)
plt.yscale('log')
plt.xlabel('Valor da Transação (Amount)')
plt.ylabel('Frequência (log)')
plt.title('Distribuição do Valor das Transações - Fraude vs. Não Fraude')
plt.legend()
plt.show()


# ##### Analisando a plotagem abaixo, podemos tirar algumas conclusões:
# A mediana de todas as transações classificadas como fraude tende a ter valores mais baixos.
# 
# O intervalo interquartil (IQR) das transações fraudulentas é maior do que o das transações não fraudulentas. Isso significa que há uma maior taxa de variação nas transações fraudulentas.
# 
# A cauda do IQR das transações fraudulentas é mais alta do que a das transações não fraudulentas. Isso indica que, embora as transações fraudulentas sejam quase sempre de valores baixos, podem ocorrer, em situações atípicas, transações com valores mais altos.
# 
# 

# In[14]:


plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Class'], y=df['Amount'], width=0.2)
plt.ylim(0, df['Amount'].quantile(0.95))  # Limita o eixo Y para remover valores muito altos e facilitar a visualização
plt.title('Distribuição do Valor das Transações - Fraude vs. Não Fraude')
plt.xlabel('Classe (0 = Não Fraude, 1 = Fraude)')
plt.ylabel('Valor da Transação')
plt.show()


# #### Verificar quantas fraudes ocorreram em valores realmente mais altos
# Interpretando o resultado do código abaixo, observamos que, de fato, a maioria das fraudes ocorre em transações de valores mais baixos, enquanto as transações não fraudulentas estão bem distribuídas entre valores baixos e altos. Ou seja, de 100% das transações fraudulentas, apenas 8,73% são de valores acima do limite do percentil 95. Isso apenas reforça que, ao escolher entre realizar uma transação fraudulenta com valores altos ou baixos, os fraudadores tendem a realizar a maioria das transações com valores baixos, a fim de não levantar suspeitas. Somente em esporádicas ocasiões, quando buscam um golpe mais arriscado, optam por transações de valores mais altos, mas com grandes possibilidades de obter um lucro maior.

# In[15]:


import numpy as np

# Definir o limite de valores altos (ex: 95º percentil das transações)
limite_alto = np.percentile(df['Amount'], 95)

# Contar fraudes acima do limite
fraudes_altas = df[(df['Class'] == 1) & (df['Amount'] > limite_alto)]
nao_fraudes_altas = df[(df['Class'] == 0) & (df['Amount'] > limite_alto)]

print(f"Total de fraudes acima do percentil 95: {len(fraudes_altas)}")
print(f"Total de transações não fraudulentas acima do percentil 95: {len(nao_fraudes_altas)}")


# #### Comparando médias dos valores das transações fraudulentas e não fraudulentas
# Como podemos ver, a média dos valores das transações fraudulentas é maior do que a das transações não fraudulentas. Como observamos anteriormente, a maioria das transações fraudulentas tem valores mais baixos, mas o valor das transações pode ser um fator relevante para identificar fraudes. No entanto, ele não pode ser utilizado como único critério. Vamos analisar mais à frente como essa informação se comporta por meio do cálculo da mediana.

# In[16]:


# Calcular a média de Amount para fraudes e não fraudes
media_fraude = df[df['Class'] == 1]['Amount'].mean()
media_nao_fraude = df[df['Class'] == 0]['Amount'].mean()

print(f'Média de Amount para fraudes: {media_fraude:.2f}')
print(f'Média de Amount para não fraudes: {media_nao_fraude:.2f}')


# #### Comparando medianas dos valores das transações fraudulentas e não fraudulentas
# Agora ficou interessante! Vemos que o comportamento da mediana confirma o que estávamos falando acima: a maioria das transações fraudulentas tem valores baixos. O cálculo da média, por outro lado, mostrou o contrário, justamente porque algumas transações fraudulentas têm valores altos, o que acaba puxando a média para cima.

# In[17]:


# Calcular a mediana de Amount para fraudes e não fraudes
media_fraude = df[df['Class'] == 1]['Amount'].median()
media_nao_fraude = df[df['Class'] == 0]['Amount'].median()

print(f'Mediana de Amount para fraudes: {media_fraude:.2f}')
print(f'Mediana de Amount para não fraudes: {media_nao_fraude:.2f}')


# #### Analisando a dispersão dos valores das transações fraudulentas e não fraudulentas
# Analisando o gráfico abaixo, vemos que existem mais pontos densos nas transações não fraudulentas do que nas fraudulentas. Como há mais transações não fraudulentas, os valores dessas transações acabam se espalhando por toda a faixa de valores. Já nas transações fraudulentas, há menos pontos e menor densidade. Isso significa que as fraudes seguem um padrão mais específico de valores, ou seja, confirma o que estamos observando sobre valores baixos na maioria das transações fraudulentas.

# In[18]:


plt.figure(figsize=(6, 4))

# Criando o boxplot + stripplot para visualizar melhor os outliers
sns.boxplot(x='Class', y='Amount', data=df, showfliers=False, width=0.5)
sns.stripplot(x='Class', y='Amount', data=df, jitter=True, alpha=0.3, color='black')

plt.yscale('log')  # Escala logarítmica para melhor visualização
plt.xlabel('Classe (0 = Não Fraude, 1 = Fraude)')
plt.ylabel('Valor da Transação (log)')
plt.title('Dispersão dos Valores das Transações - Fraude vs. Não Fraude')

plt.show()


# #### Verificando se há uma correlação linear forte entre as variáveis dependentes e a variável classe
# Analisando a correlação abaixo, vemos que não existe nenhuma variável com correlação forte com a variável classe, ou seja, todas apresentam correlações baixas. Isso significa que o modelo de regressão logística, apesar de ser um modelo que pode funcionar para um problema dessa natureza (binário), pode não conseguir aprender de forma eficaz a identificar as fraudes, justamente porque não há uma correlação linear forte entre as variáveis dependentes e a variável classe. Diante desse cenário, além da regressão logística, utilizaremos modelos como XGBoost e RandomForest, visto que esses dois últimos podem se adaptar melhor a esse conjunto de dados e capturar as fraudes com mais eficácia.

# In[19]:


# Verifique a correlação entre as variáveis independentes e a variável dependente
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm', axis=None)

# Ajustando a largura da tabela
corr.style.set_table_attributes('style="width: 70%;"').background_gradient(cmap='coolwarm', axis=None)

# Ajustando o tamanho da fonte e outras configurações
corr.style.set_table_styles(
    [{'selector': 'th', 'props': [('font-size', '10px')]},  # Tamanho da fonte das colunas
     {'selector': 'td', 'props': [('font-size', '9px')]}   # Tamanho da fonte das células
]).background_gradient(cmap='coolwarm', axis=None)


# ### Preparação dos dados para o modelo

# #### Verificando a existência de dados faltantes
# 
# É verificado que o conjunto de dados não possui dados faltantes

# In[20]:


df.isnull().sum()


# #### Verificando se existem variáveis categóricas
# 
# O conjunto de dados não possui atributos do tipo categórico

# In[21]:


df.dtypes


# #### Retirando a variável Tempo

# In[22]:


# Tentar dropar a coluna 'Time', ignorando o erro se a coluna não existir
df.drop('Time', axis=1, errors='ignore', inplace=True)
df.head()


# #### Tratamento dos outliers
# Como temos um conjunto de dados altamente desbalanceado, optamos por manter os outliers, visto que estamos tratando da identificação de fraudes e, pelas nossas análises, as fraudes também podem assumir a forma de transações atípicas. Retirá-los seria impedir que o modelo tentasse aprender sobre essas transações.

# #### Aplicação de técnicas de balanceamento
# Nosso conjunto de dados é bastante desbalanceado. Podemos até adotar técnicas de adicionar pesos maiores às observações da classe minoritária, porém, como a diferença entre as classes é muito grande, utilizaremos técnicas de balanceamento de dados já aplicadas na literatura. Como queremos identificar fraudes e a amostra de observações fraudulentas é extremamente menor, iremos aumentar a quantidade de amostras fraudulentas para melhorar a capacidade do modelo em identificá-las. Utilizaremos o Oversampling para as fraudes.

# In[23]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings

# Suprimir o aviso específico
warnings.filterwarnings("ignore", category=FutureWarning)

X = df.drop('Class', axis=1)  # Variáveis independentes
y = df['Class']  # Variável alvo (fraude ou não fraude)

smote = SMOTE(sampling_strategy='auto', random_state=42)
ovs_X, ovs_y = smote.fit_resample(X, y)

print('X', ovs_X.shape)
print('y', ovs_y.shape)


# #### Verificando a importância das variáveis
# Ao analisar o gráfico abaixo, vemos que as três variáveis mais importantes do conjunto de dados extraídas através do PCA são, na ordem, V10, V4 e V14. Iremos, mais à frente, testá-las nos modelos e avaliar o seu nível de representatividade.

# In[24]:


from sklearn.ensemble import RandomForestClassifier

# Pegar apenas 10% do conjunto de dados
X_train, X_test, y_train, y_test = train_test_split(ovs_X, ovs_y, test_size=0.9, random_state=42)

# Criar e treinar o modelo RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Obter a importância das features
feature_importance = rf.feature_importances_

# Criar um DataFrame para visualizar
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})

# Ordenar as features pela sua importância
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotando o gráfico de barras com colormap
plt.figure(figsize=(8, 6))

# Usando um colormap 'viridis' que vai de cores mais frias (menos importantes) para mais quentes (mais importantes)
plt.barh(importance_df['Feature'], importance_df['Importance'], color=plt.cm.viridis(importance_df['Importance'] / max(importance_df['Importance'])))

plt.xlabel('Importância')
plt.title('Importância das Features no Random Forest')
plt.gca().invert_yaxis()  # Inverter o gráfico para as barras mais importantes ficarem no topo
plt.show()


# #### Clusterizando o conjunto de dados após o Oversampling com as features mais importantes
# Ao realizar a clusterização com as features mais importantes após o balanceamento, podemos perceber que é possível obter uma boa margem de separação dos dados entre as classes de fraude e não fraude. Isso significa que o conjunto de dados está mais adequado para ser trabalhado em modelos supervisionados, como veremos mais à frente.
# 
# Obs: utilizamos apenas as duas features mais importantes para possibilitar uma visualização em um espaço 2D dos clusters.

# In[27]:


from sklearn.cluster import MiniBatchKMeans

#Gerando o conjunto de dados com a features mais importantes
X_cluster = ovs_X[['V10', 'V4']]
X_cluster

# Definir o número de clusters (por exemplo, k=2)
kmeans = MiniBatchKMeans(n_clusters=2, random_state=42)

# Ajustar o modelo aos dados (as duas colunas do DataFrame)
kmeans.fit(X_cluster)

# Obter os rótulos de cluster atribuídos a cada ponto
labels = kmeans.labels_

# Obter as coordenadas dos centros dos clusters
centroids = kmeans.cluster_centers_

# Plotar os dados e os clusters
plt.figure(figsize=(8, 6))

# Plotar os pontos de dados com base nos rótulos de clusters
plt.scatter(X_cluster['V10'], X_cluster['V4'], c=labels, cmap='viridis', s=30, marker='o')

# Plotar os centros dos clusters
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')

# Adicionar rótulos e título
plt.title('KMeans MiniBatch - Clusters e Centros')
plt.xlabel('Feature - V10')
plt.ylabel('Feature - V4')

# Mostrar a legenda
plt.legend()

# Exibir o gráfico
plt.show()


# #### Aplicando normalização nos dados

# In[26]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled_superv = scaler.fit_transform(ovs_X)  # X é o conjunto de variáveis independentes


# #### Treinamento análise dos resultados para os modelos

# ##### Definindo funções

# In[33]:


def cross_validation(model, X_train, y_train, folds):

    # Definindo múltiplas métricas
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    # Aplicando validação cruzada
    # cv=5 significa que estamos usando 10 folds
    scores = cross_validate(model, X_train, y_train, cv=folds, scoring=scoring)
    
    # Exibindo as pontuações de cada métrica
    print(f'Pontuação de accuracy em cada fold: {scores["test_accuracy"]}')
    print(f'Pontuação de precision em cada fold: {scores["test_precision"]}')
    print(f'Pontuação de recall em cada fold: {scores["test_recall"]}')
    print(f'Pontuação de f1 em cada fold: {scores["test_f1"]}')
    
    # Média e desvio padrão das pontuações
    print(f'Média da pontuação de accuracy: {scores["test_accuracy"].mean()}')
    print(f'Desvio padrão de accuracy: {scores["test_accuracy"].std()}')
    
    print(f'Média da pontuação de precision: {scores["test_precision"].mean()}')
    print(f'Desvio padrão de precision: {scores["test_precision"].std()}')
    
    print(f'Média da pontuação de recall: {scores["test_recall"].mean()}')
    print(f'Desvio padrão de recall: {scores["test_recall"].std()}')
    
    print(f'Média da pontuação de f1: {scores["test_f1"].mean()}')
    print(f'Desvio padrão de f1: {scores["test_f1"].std()}')


# ##### Dividindo o conjunto de dados em treino e teste (O tetse não será usado pq estaremos usando validação cruzada)

# In[38]:


# Dividindo em treino e teste (não será usado o teste na validação cruzada)
X_train, X_test, y_train, y_test = train_test_split(X_scaled_superv, ovs_y, test_size=0.2, random_state=42)


# ##### Testando o modelo LogisticRegression

# In[39]:


from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Criando o modelo
model = LogisticRegression(max_iter=1000)
cross_validation(model, X_train, y_train, 5)


# ##### Testando o modelo XGBoost

# In[41]:


import xgboost as xgb

# Criar o modelo XGBoost com os paramêtros mais utlizados na maioria dos problemas de classificação
model = xgb.XGBClassifier(
    n_estimators=100,          # Número de árvores
    learning_rate=0.1,         # Taxa de aprendizado
    max_depth=5,               # Profundidade máxima das árvores
    min_child_weight=3,        # Peso mínimo de uma folha
    subsample=0.8,             # Subamostragem
    colsample_bytree=0.9,      # Fração das colunas usadas para cada árvore
    gamma=0,                   # Redução mínima da perda
    objective='binary:logistic', # Função de perda para classificação binária
    eval_metric='logloss',     # Métrica de avaliação
)

cross_validation(model, X_train, y_train, 5)


# ##### Testando o modelo RandomForest

# In[44]:


from sklearn.ensemble import RandomForestClassifier

# Criando o modelo de RandomForest
model = RandomForestClassifier(n_estimators=50, random_state=42)
cross_validation(model, X_train, y_train, 5)


# #### Resumo e Conclusão dos Resultados dos Modelos
# Neste trabalho, foi realizado um processo de oversampling utilizando o método SMOTE para balancear o conjunto de dados, que inicialmente apresentava um desequilíbrio entre as classes de fraude e não fraude. O objetivo foi melhorar a performance dos modelos de Machine Learning na detecção de fraudes, garantindo que o modelo tivesse mais amostras da classe minoritária.
# 
# Além disso, foi realizada uma análise exploratória com o algoritmo K-means para agrupar os dados nas duas principais classes (fraude e não fraude). A análise revelou uma boa margem de separação entre os clusters, o que indica que as features escolhidas são adequadas para distinguir as duas classes, porém optamos por usar todas elas nos modelos supervisionados. O objetivo com o K-means era apenas analisar o quão fácil ou difícil ficou o conjunto de dados após o oversampling.
#  
# #### Resultados dos Modelos:
# ##### Regressão Logística:
# 
# ##### Acurácia média: 0.9561.
# 
# ##### Precision média: 0.9824.
# 
# ##### Recall médio: 0.9289.
# 
# ##### F1-Score médio: 0.9549
# 
# A Regressão Logística apresentou boa performance, mas não alcançou resultados ideais na detecção de fraudes (recall).
# ##### -------------------------------------------------------------------------------------------------------------------------------------
# 
# ##### XGBoost:
# 
# ##### Acurácia média: 0.9967.
# 
# ##### Precision média: 0.9961.
# 
# ##### Recall médio: 0.9973.
# 
# ##### F1-Score médio: 0.9967.
# 
# O modelo XGBoost mostrou um desempenho excelente, com alta precisão e recall, e um ótimo equilíbrio entre precisão e revocação.
# 
# ##### -------------------------------------------------------------------------------------------------------------------------------------
# 
# ##### Random Forest:
# 
# ##### Acurácia média: 0.9999.
# 
# ##### Precision média: 0.9998.
# 
# ##### Recall médio: 1.0.
# 
# ##### F1-Score médio: 0.9999.
# 
# O modelo Random Forest obteve os melhores resultados, especialmente em recall, com um valor de 1.0, indicando que o modelo foi capaz de identificar todas as fraudes sem falsos negativos.
# 
# #### Conclusão
# Com base nos resultados obtidos com os parâmetros selecionado a princ, o modelo Random Forest foi o mais eficaz na detecção de fraudes, superando o XGBoost e a Regressão Logística em todas as métricas de avaliação, especialmente na recall. A técnica de oversampling utilizando SMOTE foi crucial para melhorar a performance do modelo, e o K-means indicou uma boa separação entre as classes, confirmando a relevância das variáveis utilizadas.
# 
# Portanto, o Random Forest é o modelo recomendado para este problema, devido à sua capacidade de alcançar um ótimo equilíbrio entre precisão e recall, além de minimizar os falsos negativos, o que é crucial em problemas de detecção de fraude.
