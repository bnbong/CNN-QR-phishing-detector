import tensorflow as tf

from src.cnn_qr_phishing_detector.model.cnn import build_cnn_model
from src.cnn_qr_phishing_detector.preprocess.preprocessor import ImagePreprocessor
from src.cnn_qr_phishing_detector.utils import get_callbacks


class ModelTrainer:
    def __init__(self, data_dir, img_size=(128, 128), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.preprocessor = ImagePreprocessor(img_size, batch_size)

    def train(self, epochs=50):
        train_gen, val_gen = self.preprocessor.create_data_generators(self.data_dir)

        model = build_cnn_model(input_shape=(*self.img_size, 1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        csv_logger = tf.keras.callbacks.CSVLogger("training_log.csv")

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=get_callbacks() + [csv_logger],
        )

        return model, history
