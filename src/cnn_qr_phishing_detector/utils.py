import tensorflow as tf


def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU 설정 완료: {len(gpus)}개의 GPU 사용 가능")
        except RuntimeError as e:
            print(e)
    else:
        print("GPU 설정 불가, CPU로 학습")


def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True),
    ]
