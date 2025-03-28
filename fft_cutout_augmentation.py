import numpy as np
import cv2
import os
import argparse

def apply_fft_cutout(img_path, output_dir, num_images=100):
    # RGB 이미지 읽기
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 이미지 크기를 짝수로 조정
    height, width, _ = img_rgb.shape
    if height % 2 != 0:
        img_rgb = img_rgb[:-1, :, :]
    if width % 2 != 0:
        img_rgb = img_rgb[:, :-1, :]
    
    # 저장할 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 지정된 횟수만큼 반복하여 필터링 및 저장
    for i in range(num_images):
        # 랜덤 cutout 영역 설정
        height, width, _ = img_rgb.shape
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        w = np.random.randint(20, 120)
        h = np.random.randint(20, 120)
        
        # 각 채널별로 2차원 FFT 변환 및 cutout 적용
        fft_channels = []
        for channel in range(3):
            fft = np.fft.fft2(img_rgb[:, :, channel])
            fft_shift = np.fft.fftshift(fft)
            c_height, c_width = fft_shift.shape
            
            # cutout 영역이 이미지 경계를 넘지 않도록 조정
            x_end = min(x + w, c_width)
            y_end = min(y + h, c_height)
            
            # cutout 적용 (주파수 영역에서 특정 부분을 0으로 설정)
            fft_shift[y:y_end, x:x_end] = 0
            
            # 역 시프트 및 채널 저장
            fft = np.fft.ifftshift(fft_shift)
            fft_channels.append(fft)
        
        # 각 채널별로 2차원 역-FFT 변환
        filtered_channels = [np.fft.ifft2(fft).real for fft in fft_channels]
        filtered_img = np.dstack(filtered_channels)
        
        # 이미지 값 범위 조정 및 저장
        filtered_img_rgb = np.clip(filtered_img, 0, 255).astype(np.uint8)
        save_path = os.path.join(output_dir, f'filtered_image_{i+1}.png')
        cv2.imwrite(save_path, cv2.cvtColor(filtered_img_rgb, cv2.COLOR_RGB2BGR))
    
    print(f'{num_images} filtered images saved successfully to {output_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply FFT Cutout Augmentation to color images')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='./filtered_images', help='Output directory path')
    parser.add_argument('--count', type=int, default=100, help='Number of augmented images to generate')
    
    args = parser.parse_args()
    apply_fft_cutout(args.input, args.output, args.count)
