import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, MultiHeadAttention,
    LayerNormalization, Add, GlobalAveragePooling1D, Concatenate, Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    attn_out = Add()([inputs, attn_out])
    attn_out = LayerNormalization(epsilon=1e-6)(attn_out)

    ffn_out = Dense(ff_dim, activation="gelu")(attn_out)
    ffn_out = Dropout(dropout)(ffn_out)
    ffn_out = Dense(inputs.shape[-1])(ffn_out)
    ffn_out = Add()([attn_out, ffn_out])
    return LayerNormalization(epsilon=1e-6)(ffn_out)


def bidirectional_cross_modal_attention(vision_features, gene_features, num_heads=8, key_dim=64):
    proj_dim = key_dim * num_heads

    v_proj = Dense(proj_dim, name='vision_projection_bcma')(vision_features)
    g_proj = Dense(proj_dim, name='gene_projection_bcma')(gene_features)

    v2g_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name='v2g_attention')(
        query=v_proj, key=g_proj, value=g_proj
    )
    vision_attended = Add(name='v2g_add')([v_proj, v2g_out])
    vision_attended = LayerNormalization(epsilon=1e-6, name='v2g_layernorm')(vision_attended)

    g2v_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name='g2v_attention')(
        query=g_proj, key=v_proj, value=v_proj
    )
    gene_attended = Add(name='g2v_add')([g_proj, g2v_out])
    gene_attended = LayerNormalization(epsilon=1e-6, name='g2v_layernorm')(gene_attended)

    return vision_attended, gene_attended


def build_transformer_model(
    gene_dims,
    vision_feature_dim=2048,
    num_patches=50,
    num_heads=8,
    ff_dim=512,
    num_transformer_blocks=2,
):
    vision_input = Input(shape=(num_patches, vision_feature_dim), name="vision_features")
    gene_input = Input(shape=(gene_dims,), name="gene_features")

    # Gene branch
    g = Dense(512, activation="gelu", name="gene_dense_1")(gene_input)
    g = BatchNormalization(name="gene_bn_1")(g)
    g = Dropout(0.3, name="gene_dropout_1")(g)
    g = Reshape((1, 512), name="gene_expand_dims")(g)

    # Vision branch: transformer encoder blocks
    v = vision_input
    for _ in range(num_transformer_blocks):
        head_size = vision_feature_dim // num_heads if vision_feature_dim % num_heads == 0 else 64
        v = transformer_encoder_block(v, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, dropout=0.2)

    # Bidirectional cross-modal attention
    vision_attended, gene_attended = bidirectional_cross_modal_attention(
        v, g, num_heads=num_heads, key_dim=64
    )

    # Pooling and fusion
    u_v = GlobalAveragePooling1D(name="vision_avg_pool")(vision_attended)
    u_g = Reshape((64 * num_heads,), name="gene_squeeze")(gene_attended)
    combined = Concatenate(name="combine_modalities")([u_v, u_g])

    # Classification MLP
    x = Dense(512, activation="gelu", kernel_regularizer=l2(1e-4), name="combined_dense_1")(combined)
    x = BatchNormalization(name="combined_bn_1")(x)
    x = Dropout(0.5, name="combined_dropout_1")(x)
    x = Dense(256, activation="gelu", kernel_regularizer=l2(1e-4), name="combined_dense_2")(x)
    x = Dropout(0.3, name="combined_dropout_2")(x)
    output = Dense(1, activation="sigmoid", name="prediction_output")(x)

    model = Model(inputs=[vision_input, gene_input], outputs=output, name="PathGenFusion")
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    return model
