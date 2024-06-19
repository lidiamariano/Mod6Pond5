from flask import Flask, render_template, request, redirect, url_for
import os
from predict import predict

app = Flask(__name__)

UPLOAD_FOLDER = '../uploads'
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
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Passa o caminho do arquivo para a função predict
        resultado_predicao = predict(file_path)
        
        # Redireciona para a página de resultado com o número predito
        return redirect(url_for('resultado', predicao=resultado_predicao))

@app.route('/resultado')
def resultado():
    predicao = request.args.get('predicao')
    return render_template('resultado.html', predicao=predicao)

if __name__ == '__main__':
    app.run(debug=True)
