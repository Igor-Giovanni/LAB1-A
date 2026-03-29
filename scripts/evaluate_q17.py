import sys
from pathlib import Path

# Add parent directory to path so we can import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import Config

cfg = Config()

def simular_q1_7(pesos_float):
    """
    Degrada o peso do Python para os exatos degraus que a FPGA consegue ler em Q1.7.
    """
    # 1. Escala o número multiplicando por 128 (2^7)
    escala = 128.0 
    
    # 2. Arredonda e satura entre os limites do 8 bits (-128 a 127)
    pesos_quantizados = np.clip(np.round(pesos_float * escala), -128, 127)
    
    # 3. Retorna o valor dividindo por 128 (simulando a leitura do hardware)
    return pesos_quantizados / escala

def avaliar_modelo_fpga():
    print("\n[SIMULADOR FPGA] Carregando modelo em Float32...")
    model_path = cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
    model = tf.keras.models.load_model(str(model_path))
    
    print("[SIMULADOR FPGA] Degradando precisão dos pesos para Q1.7...")
    for layer in model.layers:
        pesos_da_camada = layer.get_weights()
        if not pesos_da_camada: 
            continue # Pula camadas que não têm pesos (como o MaxPooling)
        
        # weights[0] são as matrizes, weights[1] são os bias
        pesos_degradados = [simular_q1_7(w) for w in pesos_da_camada]
        layer.set_weights(pesos_degradados)
        
    print("[SIMULADOR FPGA] Pesos injetados! A rede agora tem o cérebro da FPGA.\n")
    
    # Carrega as imagens de teste
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        cfg.PROCESSED_DIR / "test",
        target_size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
        color_mode="grayscale",
        class_mode="binary",
        batch_size=32,
        shuffle=False
    )
    
    print("Gerando predições com o modelo degradado...")
    y_pred_prob = model.predict(test_gen)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    
    print("\n--- Relatório de Classificação (SIMULAÇÃO HARDWARE) ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plota e salva a Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Usando cor Laranja/Vermelha para diferenciar do gráfico original (Azul)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito pela FPGA Simualada')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão - Hardware Ponto Fixo (Q1.7)')
    
    img_path = cfg.PROJECT_ROOT / "reports" / "matriz_confusao_q17.png"
    plt.savefig(str(img_path))
    print(f"\nNova Matriz de Confusão salva em: {img_path}")
    plt.show()

if __name__ == "__main__":
    avaliar_modelo_fpga()