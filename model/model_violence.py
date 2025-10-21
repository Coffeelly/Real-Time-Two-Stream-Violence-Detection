import tensorflow as tf
from tensorflow.keras import layers, Model

# ==== Konstanta Model ====
FRAMES_PER_SEQUENCE = 16
VIDEO_HEIGHT = 96
VIDEO_WIDTH = 96
N_CHANNELS = 3
FUSION_FILTERS = 32

def build_spatial_extractor_block(name_prefix="sepconv_block"):
    """Membangun blok CNN untuk ekstraksi fitur spasial."""
    block = tf.keras.Sequential([
        layers.SeparableConv2D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2)
    ], name=name_prefix)
    return block

def build_violence_model(model_weights_path):
    """
    Membangun arsitektur model deteksi kekerasan dan me-load bobotnya.
    """
    # ==== Input Layer ====
    inputs_raw = layers.Input(shape=(FRAMES_PER_SEQUENCE, VIDEO_HEIGHT, VIDEO_WIDTH, N_CHANNELS), name="input_raw")
    inputs_skel = layers.Input(shape=(FRAMES_PER_SEQUENCE, VIDEO_HEIGHT, VIDEO_WIDTH, N_CHANNELS), name="input_skel")

    # ==== Normalisasi ====
    norm_raw = layers.TimeDistributed(layers.Lambda(lambda x: x / 255.0), name="normalize_raw")(inputs_raw)
    norm_skel = layers.TimeDistributed(layers.Lambda(lambda x: x / 255.0), name="normalize_skel")(inputs_skel)

    # ==== Ekstraksi Fitur Spasial ====
    extractor = build_spatial_extractor_block()
    raw_features = layers.TimeDistributed(extractor, name="raw_spatial_features")(norm_raw)
    skel_features = layers.TimeDistributed(extractor, name="skel_spatial_features")(norm_skel)

    # ==== Cabang 1: ConvLSTM untuk frame diff ====
    raw_bn = layers.BatchNormalization(name="bn_raw")(raw_features)
    raw_temporal = layers.ConvLSTM2D(FUSION_FILTERS, kernel_size=3, padding='same', return_sequences=True,
                                     activation='tanh', name="conv_lstm_raw")(raw_bn)

    # ==== Cabang 2: Conv + BN untuk skeleton ====
    skel_conv = layers.TimeDistributed(
        layers.Conv2D(FUSION_FILTERS, kernel_size=3, padding='same', activation='relu'),
        name="skel_conv"
    )(skel_features)
    skel_bn = layers.BatchNormalization(scale=False, center=False, name="bn_skel")(skel_conv)

    # ==== Fusi ====
    fused = layers.Add(name="fusion_add")([raw_temporal, skel_bn])

    # ==== Final Layers ====
    x = layers.ConvLSTM2D(32, kernel_size=3, return_sequences=False, padding='same', activation='tanh')(fused)
    x = layers.DepthwiseConv2D(3, depth_multiplier=2, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # ==== Bangun Model ====
    model = Model(inputs=[inputs_raw, inputs_skel], outputs=outputs)
    
    # Load bobot yang sudah dilatih
    print(f"[INFO] Loading model weights from: {model_weights_path}")
    model.load_weights(model_weights_path)
    print("[INFO] Model weights loaded successfully.")
    
    return model