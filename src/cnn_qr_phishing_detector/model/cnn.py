import tensorflow as tf


def build_cnn_model(input_shape=(128, 128, 1)):
    model = tf.keras.models.Sequential()

    model.add(
        tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", activation="relu", input_shape=input_shape
        )
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # FC
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation="softmax"))

    return model
