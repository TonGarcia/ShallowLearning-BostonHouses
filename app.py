import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# função para carregar o dataset
# @st.cache = salvar o resultado desta operação em cache para acelerar a velocidade de carregamento do app
@st.cache
def get_data():
    return pd.read_csv("data/data.csv")


# função para treinar o modelo
def train_model():
    data = get_data()
    x = data.drop("MEDV",axis=1)
    y = data["MEDV"]
    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7, max_features=3)
    rf_regressor.fit(x, y)
    return rf_regressor

# criando um dataframe
data = get_data()

# treinando o modelo
model = train_model()

# título da aplicação
st.title("DataApp - Prevendo Valores de Imóveis")

# subtítulo da aplicação
st.markdown("Este é um DataApp utilizado para exibir a solução de Machine Learning para o problema de predição de valores de imóveis de Boston.")

# verificando o dataset
st.subheader("Selecionando apenas um pequeno conjunto de atributos")

# atributos para serem exibidos por padrão
# seleção de atributos a serem exibidos no DataApp
defaultcols = ["RM","PTRATIO","LSTAT","MEDV"]

# defindo atributos a partir do multiselect
# criação de um seletor na tela do DataApp
#--> o param default é aonde passamos as colunas que queremos exibir, assim ele não exibe todos as colunas
cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)

# exibindo os top 10 registro do dataframe
#aqui cria uma seção no DataApp que exibe 10 registros
st.dataframe(data[cols].head(10))
#-> cria mais um "H2" na tela do DataApp
st.subheader("Distribuição de imóveis por preço")

# definindo a faixa de valores
#--> cria um slider para fazer filtragem dos dados na view do DataApp para selecionar o range de preços
faixa_valores = st.slider("Faixa de preço", float(data.MEDV.min()), 150., (10.0, 100.0))

# filtrando os dados
#--> cria a filtragem em cima do valor recebido do slider para retornar imóveis segundo a seleção do usuário
dados = data[data['MEDV'].between(left=faixa_valores[0],right=faixa_valores[1])]

# plot a distribuição dos dados
#--> plot dinâmico, a medida que vai alterando os filtros ele já altera a saída pro usuário
f = px.histogram(dados, x="MEDV", nbins=100, title="Distribuição de Preços")
f.update_xaxes(title="MEDV")
f.update_yaxes(title="Total Imóveis")
st.plotly_chart(f)

#--> aqui inicia-se outra sessão do app, nesta estamos criando a sessão para a predição
#--> como criou um subheader na sidebar ele cria este elemento no lado esquerdo do DataApp
st.sidebar.subheader("Defina os atributos do imóvel para predição")

# mapeando dados do usuário para cada atributo
#--> criação dos inputs editáveis para o usuário na sidebar
#cria inputs com btn, label e val default no sidebar
crim = st.sidebar.number_input("Taxa de Criminalidade", value=data.CRIM.mean())
indus = st.sidebar.number_input("Proporção de Hectares de Negócio", value=data.CRIM.mean())
#para criar opções do select é usado tupla ao invés de lista
chas = st.sidebar.selectbox("Faz limite com o rio?",("Sim","Não"))

# transformando o dado de entrada em valor binário
#--> o select retorna o valor passado como opções acima, nesse ternário converte para booleano
chas = 1 if chas == "Sim" else 0

#--> inputs de outros campos a serem simulados por entradas de usuário
nox = st.sidebar.number_input("Concentração de óxido nítrico", value=data.NOX.mean())
rm = st.sidebar.number_input("Número de Quartos", value=1)
ptratio = st.sidebar.number_input("Índice de alunos para professores",value=data.PTRATIO.mean())
b = st.sidebar.number_input("Proporção de pessoas com descendencia afro-americana",value=data.B.mean())
lstat = st.sidebar.number_input("Porcentagem de status baixo",value=data.LSTAT.mean())

# inserindo um botão na tela
#--> cria btn com ajax preparado
btn_predict = st.sidebar.button("Realizar Predição")

# verifica se o botão foi acionado
#-->ação do btn ao ser clicado no ajax
if btn_predict:
    # roda a predição
    result = model.predict([[crim,indus,chas,nox,rm,ptratio,b,lstat]])
    # cria uma seção com um H2
    st.header("O valor previsto para o imóvel é:")
    # cria um paragrafo com o valor da predição concatenado
    str_currency_result = "${:,.2f}".format((round(result[0]*10,2)*1000))
    result = "US"+str_currency_result
    st.write(result)
