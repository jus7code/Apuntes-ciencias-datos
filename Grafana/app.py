# App web para predecir dígitos escritos a mano con la CNN del Cuaderno 8 A2.
# Uso local:            pip install -r requirements.txt  &&  python app.py
# Hugging Face Spaces:  subir app.py, requirements.txt y numerosC2.keras (SDK: Gradio)
import numpy as np
import cv2
import tensorflow as tf
import gradio as gr

clases = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
modelo_cnn = tf.keras.models.load_model('numerosC2.keras')

def lienzo_a_gris(dibujo):
    """Convierte la salida del Sketchpad en una imagen en grises con trazo blanco y fondo negro (0-255)."""
    if isinstance(dibujo, dict):            # Gradio 4/5 entrega {'background', 'layers', 'composite'}
        dibujo = dibujo.get('composite')
    if dibujo is None:
        return None
    img = np.asarray(dibujo)
    if img.ndim == 3 and img.shape[2] == 4:
        alfa = img[..., 3].astype(np.float32)
        if alfa.min() < 250:                # fondo transparente: el trazo está en el canal alfa
            return alfa
        img = img[..., :3]
    if img.ndim == 3:
        img = img.mean(axis=2)
    gris = img.astype(np.float32)
    if gris.mean() > 127:                   # fondo blanco con trazo oscuro: invertir
        gris = 255.0 - gris
    return gris


def preprocesar_dibujo(dibujo):
    """Devuelve un tensor (1, 28, 28, 1) con valores en [0, 1], o None si el lienzo está vacío."""
    gris = lienzo_a_gris(dibujo)
    if gris is None:
        return None
    ys, xs = np.nonzero(gris > 30)
    if len(xs) == 0:
        return None

    # 1) Recortar el rectángulo que contiene el número
    recorte = gris[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 2) Escalar para que el lado mayor mida 20 px (conservando la proporción)
    alto, ancho = recorte.shape
    escala = 20.0 / max(alto, ancho)
    nuevo_ancho, nuevo_alto = max(1, round(ancho * escala)), max(1, round(alto * escala))
    recorte = cv2.resize(recorte, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

    # 3) Pegar en un lienzo negro de 28x28
    imagen = np.zeros((28, 28), dtype=np.float32)
    y0, x0 = (28 - nuevo_alto) // 2, (28 - nuevo_ancho) // 2
    imagen[y0:y0 + nuevo_alto, x0:x0 + nuevo_ancho] = recorte

    # 4) Centrar por centro de masa
    momentos = cv2.moments(imagen)
    if momentos['m00'] > 0:
        cx = momentos['m10'] / momentos['m00']
        cy = momentos['m01'] / momentos['m00']
        desplazamiento = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        imagen = cv2.warpAffine(imagen, desplazamiento, (28, 28))

    # 5) Normalizar a [0, 1] y añadir dimensiones de lote y canal
    imagen = imagen / max(imagen.max(), 1e-6)
    return imagen.reshape(1, 28, 28, 1)


def predecir(dibujo):
    entrada = preprocesar_dibujo(dibujo)
    if entrada is None:
        return {}, None
    probabilidades = modelo_cnn.predict(entrada, verbose=0)[0]
    resultado = {clases[i]: float(probabilidades[i]) for i in range(10)}
    vista_red = (entrada[0, :, :, 0] * 255).astype(np.uint8)
    vista_red = cv2.resize(vista_red, (140, 140), interpolation=cv2.INTER_NEAREST)
    return resultado, vista_red


lienzo = gr.Sketchpad(
    label='Escribe un dígito (0-9)',
    type='numpy',
    image_mode='RGBA',
    canvas_size=(280, 280),
    brush=gr.Brush(default_size=18, colors=['#000000'], color_mode='fixed'),
)

demo = gr.Interface(
    fn=predecir,
    inputs=lienzo,
    outputs=[
        gr.Label(num_top_classes=3, label='Predicción de la CNN'),
        gr.Image(label='Lo que ve la red (28x28)', height=140, width=140),
    ],
    live=True,
    title='Reconocimiento de dígitos escritos a mano con una CNN',
    description='Dibuja un número del 0 al 9 en el lienzo. La red neuronal convolucional entrenada con MNIST predice qué dígito es.',
)

if __name__ == '__main__':
    demo.launch()
