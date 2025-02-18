import cv2
import numpy as np
from PIL import Image

def merge_heatmap_with_background(heatmap_path, background_path, output_path, alpha=0.2):
    """히트맵과 배경 이미지를 합성하는 함수
    
    Args:
        heatmap_path (str): 히트맵 이미지 경로
        background_path (str): 배경 이미지 경로
        output_path (str): 출력 이미지 저장 경로
        alpha (float): 배경 이미지의 투명도 (0.0 ~ 1.0)
    """
    # 히트맵 이미지 로드
    heatmap = cv2.imread(heatmap_path)
    
    # 배경 이미지 로드 및 히트맵 크기에 맞게 리사이즈
    background = cv2.imread(background_path)
    background = cv2.resize(background, (heatmap.shape[1], heatmap.shape[0]))
    
    # 이미지 합성
    merged = cv2.addWeighted(background, alpha, heatmap, 1.0, 0)
    
    # 결과 저장
    cv2.imwrite(output_path, merged)
    print(f"합성된 이미지가 {output_path}에 저장되었습니다.")


heatmap_path = 'heatmap_only.png'
background_path = 'background.png'
output_path = 'merged_heatmap.png'

merge_heatmap_with_background(
    heatmap_path=heatmap_path,
    background_path=background_path,
    output_path=output_path,
    alpha=0.2  # 배경 이미지의 투명도 (0.2 = 20% 불투명)
)