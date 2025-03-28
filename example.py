"""
FFT Cutout Augmentation Example Script
======================================

이 스크립트는 FFT Cutout Augmentation을 사용하는 간단한 예제입니다.
샘플 이미지가 있다면 해당 이미지에 FFT Cutout을 적용하고 결과를 보여줍니다.
샘플 이미지가 없다면 간단한 테스트 이미지를 생성하여 처리합니다.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from fft_cutout_augmentation import apply_fft_cutout

def create_test_image(size=256):
    """간단한 테스트 이미지 생성"""
    # 체스보드 패턴 생성
    checkerboard = np.zeros((size, size, 3), dtype=np.uint8)
    square_size = size // 8
    
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                y1, y2 = i * square_size, (i + 1) * square_size
                x1, x2 = j * square_size, (j + 1) * square_size
                checkerboard[y1:y2, x1:x2] = [200, 0, 0]  # 빨간색
            else:
                y1, y2 = i * square_size, (i + 1) * square_size
                x1, x2 = j * square_size, (j + 1) * square_size
                checkerboard[y1:y2, x1:x2] = [0, 200, 0]  # 초록색
                
    # 대각선 패턴 추가
    for i in range(size):
        thickness = 3
        cv2.line(checkerboard, (0, i), (i, 0), (0, 0, 255), thickness)
        cv2.line(checkerboard, (size-i-1, 0), (size-1, i), (255, 255, 0), thickness)
    
    return checkerboard

def display_fft_result(original_img, output_dir):
    """원본과 FFT Cutout 적용 결과 시각화"""
    plt.figure(figsize=(15, 10))
    
    # 원본 이미지 표시
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # FFT Cutout 결과 표시 (5개까지)
    result_files = os.listdir(output_dir)
    result_files = [f for f in result_files if f.startswith('filtered_image_')]
    result_files = sorted(result_files)[:5]  # 최대 5개까지만 표시
    
    for i, file_name in enumerate(result_files):
        img_path = os.path.join(output_dir, file_name)
        img = cv2.imread(img_path)
        
        plt.subplot(2, 3, i+2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'FFT Cutout Result {i+1}')
        plt.axis('off')
    
    # 원본 이미지의 FFT 시각화
    plt.subplot(2, 3, 6)
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    plt.imshow(magnitude_spectrum, cmap='viridis')
    plt.title('FFT Magnitude Spectrum (Original)')
    plt.colorbar(shrink=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fft_cutout_results.png'))
    plt.show()
    
    print(f"결과 이미지가 {output_dir} 디렉토리에 저장되었습니다.")
    print(f"시각화 결과가 {os.path.join(output_dir, 'fft_cutout_results.png')}에 저장되었습니다.")

if __name__ == "__main__":
    # 출력 디렉토리 설정
    output_dir = './fft_cutout_results'
    
    # 샘플 이미지 경로 (없으면 테스트 이미지 생성)
    sample_path = './sample.jpg'
    
    if os.path.exists(sample_path):
        # 샘플 이미지가 있는 경우 사용
        original_img = cv2.imread(sample_path)
        print(f"샘플 이미지 {sample_path}를 사용합니다.")
    else:
        # 테스트 이미지 생성 및 저장
        original_img = create_test_image(size=256)
        cv2.imwrite('sample.jpg', original_img)
        sample_path = './sample.jpg'
        print("샘플 이미지가 없어 테스트 이미지를 생성했습니다: sample.jpg")
    
    # FFT Cutout 적용
    print("FFT Cutout 증강 적용 중...")
    apply_fft_cutout(sample_path, output_dir, num_images=5)
    
    # 결과 시각화
    display_fft_result(original_img, output_dir)
