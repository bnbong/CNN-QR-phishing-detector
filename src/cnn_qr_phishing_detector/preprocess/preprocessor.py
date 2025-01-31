import os

import tensorflow as tf


class ImagePreprocessor:
    def __init__(self, img_size=(128, 128), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size

    def create_data_generators(self, data_dir):
        # PNG 파일만 처리
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255, validation_split=0.2, dtype="float32"
        )

        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
            color_mode="grayscale",  # 흑백
            classes=["benign", "phishing"],
        )

        val_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
            color_mode="grayscale",
            classes=["benign", "phishing"],
        )

        return train_generator, val_generator
