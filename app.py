import streamlit as st
import numpy as np
import joblib
from streamlit_mnist_canvas import st_mnist_canvas
from PIL import Image

st.set_page_config(
    page_title="MNIST Classificador Binário",
    page_icon="✍️",
    layout="wide"
)

@st.cache_resource
def carregar_modelo():
    """
    Carrega o modelo pré-treinado do arquivo.
    O decorator @st.cache_resource garante que o modelo seja carregado apenas uma vez.
    """
    modelo = joblib.load('sgd_model.joblib')
    return modelo

st.title("✍️ Validador de Dígito 5 com MNIST")
st.markdown("---")
st.markdown("""
Esta aplicação utiliza um `SGDClassifier` pré-treinado para determinar se o dígito que você desenha é um **'5'** ou não.
Desenhe um único dígito no quadro abaixo e clique em **'Prever'**.
""")

modelo = carregar_modelo()
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Desenhe o dígito aqui:")
    canvas_result = st_mnist_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        key="mnist_canvas",
    )

with col2:
    st.subheader("Resultado da Previsão:")

    if st.button("Prever"):
        if canvas_result.resized_grayscale_array is not None:
            imagem_desenhada = canvas_result.resized_grayscale_array.reshape(1, 784) / 255.0
            previsao = modelo.predict(imagem_desenhada)
            
            if previsao[0]: 
                st.success("### É o dígito 5! 🎉")
            else: 
                st.error("### Não é o dígito 5. ❌")
    
            st.write("Imagem analisada (redimensionada para 28x28 pixels):")
            st.image(canvas_result.resized_grayscale_array, width=150)
        else:
            st.warning("Por favor, desenhe um dígito antes de prever.")

st.markdown("---")
st.write("Desenvolvido para a disciplina de Aprendizado de Máquina.")
