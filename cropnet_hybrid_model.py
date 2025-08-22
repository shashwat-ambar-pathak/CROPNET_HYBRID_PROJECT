import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization,
    Input, Add, GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model

def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    return x

def depthwise_block(x, filters):
    x = SeparableConv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x

def build_cropnet_hybrid(input_shape=(224, 224, 3), num_classes=38):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)

    x = depthwise_block(x, 64)
    x = residual_block(x, 64)

    x = depthwise_block(x, 128)
    x = residual_block(x, 128)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="CropNet_Hybrid")
    return model
