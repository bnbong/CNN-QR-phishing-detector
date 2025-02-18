from PIL import Image
import cairosvg
import io
import numpy as np

def convert_svg_to_png_with_padding(svg_path, output_path, padding=40, background_color='white'):
    """SVG 파일을 PNG로 변환하고 지정된 여백을 추가하는 함수
    
    Args:
        svg_path (str): 입력 SVG 파일 경로
        output_path (str): 출력 PNG 파일 경로
        padding (int): 추가할 여백 픽셀 수
        background_color (str): 배경색
    """
    # SVG를 PNG로 변환
    png_data = cairosvg.svg2png(url=svg_path)
    
    # 바이트 데이터를 PIL Image로 변환
    image = Image.open(io.BytesIO(png_data))
    
    # 새로운 크기 계산 (패딩 포함)
    new_width = image.width + (2 * padding)
    new_height = image.height + (2 * padding)
    
    # 패딩이 포함된 새 이미지 생성
    padded_image = Image.new('RGB', (new_width, new_height), background_color)
    
    # 원본 이미지를 패딩이 적용된 위치에 붙여넣기
    padded_image.paste(image, (padding, padding))
    
    # 결과 저장
    padded_image.save(output_path, 'PNG')
    print(f"변환된 이미지가 {output_path}에 저장되었습니다.")


svg_path = 'QR_code_for_mobile_English_Wikipedia.svg'
output_path = 'background.png'

convert_svg_to_png_with_padding(
    svg_path=svg_path,
    output_path=output_path,
    padding=40,
    background_color='white'
)