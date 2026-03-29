import mlflow
import mlflow.keras
import keras_tuner as kt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ModelEngine:
    """
    Motor de treino otimizado para a Tiny-CNN.
    Implementa pesos manuais e busca de hiperparâmetros focada no RECALL (detecção de autorizados).
    """
    def __init__(self, config, model_builder):
        self.cfg = config
        self.model_builder = model_builder
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment("Fechadura_Biometrica_Otimizada")

    def get_generators(self):
        """Prepara os geradores de dados com augmentation dinâmico para iluminação."""

        # OTIMIZAÇÃO: Augmentation em tempo real apenas para o TREINO
        # brightness_range: 0.5 (muito escuro) até 1.5 (muito claro/estourado)
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            brightness_range=[0.5, 1.5],
            rotation_range=10, # Pequena inclinação extra
            zoom_range=0.1     # Leve variação de distância da câmera
        )

        # Validação NUNCA deve ter augmentation, apenas rescale
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        train = train_datagen.flow_from_directory(
            self.cfg.PROCESSED_DIR / 'train',
            target_size=(32, 32),
            color_mode="grayscale",
            class_mode="binary",
            batch_size=32,
            shuffle=True
        )

        val = val_datagen.flow_from_directory(
            self.cfg.PROCESSED_DIR / 'val',
            target_size=(32, 32),
            color_mode="grayscale",
            class_mode="binary",
            batch_size=32,
            shuffle=False
        )
        return train, val
    '''def train(self):
        train_gen, val_gen = self.get_generators()

        # Classe 0 (Desconhecido): Peso 1.0
        # Classe 1 (Autorizado): Peso 4.0
        cw = {0: 1.0, 1: 4.0}

        # ALTERAÇÃO: Otimizar para maximizar o Recall na Validação
        tuner = kt.Hyperband(
            self.model_builder,
            objective=kt.Objective('val_recall', direction='max'), # Foco na detecção de autorizados
            max_epochs=20,
            factor=3,
            directory='tuner_logs',
            project_name='tiny_cnn_fine_tuning'
        )

        with mlflow.start_run(run_name="Optimized_Training_Recall"):
            stop_search = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10
            )

            print("\n[PASSO 4.1] Iniciando busca (Objetivo: MAX val_recall)...")
            tuner.search(train_gen, validation_data=val_gen, callbacks=[stop_search])

            best_hps = tuner.get_best_hyperparameters()[0]
            model = tuner.hypermodel.build(best_hps)

            # Registo automático no MLflow
            mlflow.keras.autolog(log_models=True)

            print("\n[PASSO 4.2] Iniciando ajuste fino do modelo final...")
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=50,
                class_weight=cw,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                )]
            )

            model_path = self.cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
            model.save(str(model_path))
            print(f" -> Modelo final guardado em: {model_path}")
        return history, model
'''
    def train(self):
        train_gen, val_gen = self.get_generators()

        # --- CORREÇÃO 1: CÁLCULO DINÂMICO DE PESOS ---
        # Em vez de fixar 1.0 e 4.0, vamos deixar a matemática equilibrar os seus novos vídeos
        from sklearn.utils import class_weight
        labels = train_gen.classes
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        cw = dict(enumerate(weights))
        print(f"\n[CORREÇÃO] Pesos balanceados calculados: {cw}")

        # --- CORREÇÃO 2: MUDAR O OBJETIVO DO TUNER ---
        # Focar em 'val_accuracy' impede que a rede libere todo mundo só para inflar o Recall
        tuner = kt.Hyperband(
            self.model_builder,
            objective='val_accuracy', 
            max_epochs=20,
            factor=3,
            directory='tuner_logs',
            project_name='tiny_cnn_fine_tuning_v2' # Mudei o nome para ele não usar os testes antigos
        )

        with mlflow.start_run(run_name="Optimized_Training_Accuracy"):
            stop_search = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10
            )

            print("\n[PASSO 4.1] Iniciando busca (Objetivo: MAX val_accuracy)...")
            tuner.search(train_gen, validation_data=val_gen, callbacks=[stop_search])

            best_hps = tuner.get_best_hyperparameters()[0]
            model = tuner.hypermodel.build(best_hps)

            mlflow.keras.autolog(log_models=True)

            print("\n[PASSO 4.2] Iniciando ajuste fino do modelo final...")
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=50,
                class_weight=cw, # Usando os pesos dinâmicos aqui
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                )]
            )

            model_path = self.cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5"
            model.save(str(model_path))
            print(f" -> Modelo final guardado em: {model_path}")

        return history, model
        