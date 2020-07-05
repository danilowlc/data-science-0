#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# In[4]:


df = black_friday[['Product_Category_2', 'Product_Category_3']].isna()
df.query('Product_Category_2 == True')['Product_Category_3'].all()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[5]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[6]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return int(black_friday['Gender'][black_friday['Age'] == '26-35'].value_counts()[1])


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[7]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return int(black_friday['User_ID'].nunique())


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[8]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return int(black_friday.dtypes.nunique())


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[9]:


def q5():
    df = black_friday.isna().any(1)
    return float(df.sum()/len(df))


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[10]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return int(black_friday.isnull().sum().max())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[11]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday['Product_Category_3'].value_counts().index[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[12]:


def q8():
    # Retorne aqui o resultado da questão 8.
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler().fit(np.array(black_friday['Purchase']).reshape(-1,1))
    black_friday['Purchase'] = scaler.transform(np.array(black_friday['Purchase']).reshape(-1,1))
    
    return black_friday['Purchase'].mean()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[13]:


def q9():
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(np.array(black_friday['Purchase']).reshape(-1,1))
    black_friday['Purchase'] = scaler.transform(np.array(black_friday['Purchase']).reshape(-1,1))
    return black_friday.query("Purchase >= -1 & Purchase <= 1").shape[0]


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[14]:


def q10():
    df = black_friday[['Product_Category_2', 'Product_Category_3']].isna()
    return bool(df.query('Product_Category_2 == True')['Product_Category_3'].all())


# In[ ]:




