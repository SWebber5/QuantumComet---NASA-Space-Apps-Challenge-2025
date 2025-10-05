import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics

def build_cnn_global_local_model(hp):
    """
    Dual-input 1D CNN model for Kepler-style exoplanet classification.
    """

    # ----- Input layers -----
    global_input = keras.Input(shape=(2001, 1), name="global_input")
    local_input = keras.Input(shape=(201, 1), name="local_input")

    # ----- Global view CNN branch -----
    xg = global_input
    for i in range(hp.Int("global_conv_blocks", 1, 3)):
        xg = layers.Conv1D(
            filters=hp.Choice("global_filters", [4, 16, 32]) * (i + 1),
            kernel_size=hp.Choice("global_kernel_size", [3, 5, 7]),
            activation="relu",
            padding="same"
        )(xg)
        xg = layers.MaxPooling1D(pool_size=2)(xg)
    xg = layers.Flatten()(xg)
    xg = layers.Dense(hp.Choice("global_dense_units", [32, 64, 128]), activation="relu")(xg)
    xg = layers.Dropout(hp.Float("dropout_rate", 0.2, 0.5, step=0.1))(xg)

    # ----- Local view CNN branch -----
    xl = local_input
    for i in range(hp.Int("local_conv_blocks", 1, 3)):
        xl = layers.Conv1D(
            filters=hp.Choice("local_filters", [2, 4, 8]) * (i + 1),
            kernel_size=hp.Choice("local_kernel_size", [3, 5]),
            activation="relu",
            padding="same"
        )(xl)
        xl = layers.MaxPooling1D(pool_size=2)(xl)
    xl = layers.Flatten()(xl)
    xl = layers.Dense(hp.Choice("local_dense_units", [16, 32, 64]), activation="relu")(xl)
    xl = layers.Dropout(hp.Float("dropout_rate", 0.2, 0.5, step=0.1))(xl)

    # ----- Combine both -----
    combined = layers.Concatenate()([xg, xl])
    x = layers.Dense(hp.Choice("combined_units", [64, 128, 256]), activation="relu")(combined)
    x = layers.Dropout(hp.Float("dropout_rate", 0.2, 0.5, step=0.1))(x)

    # ----- Output -----
    output = layers.Dense(1, activation="sigmoid")(x)

    # ----- Model -----
    model = keras.Model(inputs=[global_input, local_input], outputs=output)

    # ----- Compile -----
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Recall(name='recall'),
            metrics.Precision(name='precision')
            ]
    )

    return model