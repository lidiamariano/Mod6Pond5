import cv2
import numpy as np
from tensorflow.keras.models import load_model

modelo_2 = load_model('modelo_mnist.h5')

def predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img / img.max()
    _, img = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY)
    img = img.reshape(1, 28, 28, 1)
    predicao = modelo_2.predict(img)
    return str(np.argmax(predicao))
