import os

from src.cnn_qr_phishing_detector.train.train import ModelTrainer
from src.cnn_qr_phishing_detector.utils import setup_gpu


def main():
    # GPU 설정
    setup_gpu()

    # 데이터 디렉토리 확인
    data_dir = "data/images"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 학습 실행
    trainer = ModelTrainer(data_dir=data_dir)
    model, history = trainer.train(epochs=50)

    # 모델 저장
    model.save("qr_code_classifier.h5")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
