from tensorflow.keras import layers, models

def make_model():
    # Inputs
    input_global = layers.Input(shape=(2001, 1), name='global_view')  # add channel dimension
    input_local  = layers.Input(shape=(201, 1), name='local_view')

    # Global branch
    xg = layers.Conv1D(4, kernel_size=5, activation='relu')(input_global)
    xg = layers.MaxPooling1D(pool_size=2)(xg)
    # xg = layers.Conv1D(64, kernel_size=3, activation='relu')(xg)
    # xg = layers.MaxPooling1D(pool_size=2)(xg)
    xg = layers.Flatten()(xg)
    xg = layers.Dense(64, activation='relu')(xg)

    # Local branch
    xl = layers.Conv1D(2, kernel_size=3, activation='relu')(input_local)
    xl = layers.MaxPooling1D(pool_size=2)(xl)
    xl = layers.Flatten()(xl)
    xl = layers.Dense(32, activation='relu')(xl)

    # Merge branches
    x = layers.concatenate([xg, xl])
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)  # binary classification

    # Build and compile model
    model = models.Model(inputs=[input_global, input_local], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model