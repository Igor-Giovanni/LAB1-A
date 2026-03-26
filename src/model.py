import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_tiny_cnn(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(32, 32, 1)))

    # Camada Convolucional (Fixa em 4 filtros conforme requisito da FPGA)
    model.add(layers.Conv2D(
        filters=4,
        kernel_size=(3, 3),
        activation='relu',
        padding='valid',
        kernel_initializer='he_normal',
        name='conv2d_hardware'
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())

    # HIPERPARÂMETRO 1: Taxa de Dropout (Previne decoreba de pixels)
    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    model.add(layers.Dropout(rate=hp_dropout))

    # HIPERPARÂMETRO 2: Neurônios na Camada Densa
    hp_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    model.add(layers.Dense(units=hp_units, activation='relu', kernel_initializer='he_normal'))

    model.add(layers.Dense(1, activation='sigmoid'))

    # HIPERPARÂMETRO 3: Escolha do Otimizador
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])

    # HIPERPARÂMETRO 4: Taxa de Aprendizado
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    if hp_optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=hp_lr)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=hp_lr)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model