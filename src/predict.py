import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

modelo_2 = load_model('modelo_mnist.h5')

# Usa o modelo para realizar uma predição
# img = cv2.imread('/home/lidia/Mod6Pond5/static/IMG_20240606_151530.jpg')

def predict(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # predicao = modelo_2.predict(img.reshape(1, 28, 28, 1))
    img = cv2.resize(img, (28,28))
    img = img / img.max()
    _, img = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY)
    plt.imshow(img, cmap="gray_r")
    img = img.reshape(1,28,28,1)
    predicao = modelo_2.predict(img)
    print(predicao)
    # Exibe a classe com a maior probabilidade de ser a correta da predição
    np.argmax(predicao)

    return predicao
