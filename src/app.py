from flask import Flask, render_template, request
import os
from predict import predict
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'Nenhum arquivo selecionado!'
    file = request.files['file']
    if file.filename == '':
        return 'Nenhum arquivo selecionado!'
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Passa o caminho do arquivo para a função predict
        resultado_predicao = predict(file_path)
        
        return f'Arquivo enviado com sucesso! Resultado da predição: {resultado_predicao}'

@app.route('/predict/<img>', methods=['GET'])
def realizar_predicao(img):
    # Caminho para a imagem
    caminho_imagem = os.path.join(app.config['UPLOAD_FOLDER'], img)
    
    # Carrega a imagem e realiza a predição usando a função predict
    resultado_predicao = predict(caminho_imagem)
    
    return f'Resultado da predição: {resultado_predicao}'

if __name__ == '__main__':
    app.run(debug=True)
