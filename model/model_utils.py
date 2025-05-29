from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=4, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(dropout)(x)
    x = Dense(20, activation="relu")(x)
    outputs = Dense(1, activation="linear")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
