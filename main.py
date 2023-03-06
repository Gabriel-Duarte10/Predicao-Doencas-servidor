from sklearn import utils
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/calcular', methods=['POST'])
def calcular():

  data = pd.read_csv('Training.csv')
  X = data.drop(['prognosis','Unnamed: 133'], axis=1)
  
  modelo_carregado = joblib.load('modelo.pkl')
  dados = request.json
  sintomas = dados['sintomas']
  sintomas = sintomas.split(',')
  
  sintomasPred = [];
  # cria um dataframe com os sintomas inseridos
  for index, s in enumerate(X):
    sintomasPred.append(0)
    for n in sintomas:
       if n == s:
         sintomasPred[index] = 1;
  df_sintomas = pd.DataFrame([sintomasPred], columns=X.columns)
  y_pred = modelo_carregado.predict(df_sintomas)
  value = y_pred[0]
  resultado = {"result": value}
  return jsonify(resultado)

if __name__ == '__main__':
  app.run('0.0.0.0')
