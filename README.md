# FFT Cutout Augmentation for Color Images

이 저장소는 이미지의 주파수 도메인에서 Cutout 기반 데이터 증강 기법을 구현한 코드를 포함하고 있습니다. 푸리에 변환(FFT)을 사용하여 이미지를 주파수 영역으로 변환하고, 무작위 영역을 제거(cutout)한 후 다시 역변환하여 새로운 이미지를 생성합니다.

## 기능

- 컬러 이미지(RGB)를 각 채널별로 FFT 변환
- 주파수 도메인에서 무작위 위치에 Cutout 적용
- 역 FFT를 통해 변형된 이미지 생성
- 다양한 크기와 위치의 Cutout으로 여러 증강 이미지 생성

## 작동 방식

1. 입력 이미지를 RGB 채널로 분리
2. 각 채널에 2D FFT(Fast Fourier Transform) 적용
3. 주파수 도메인에서 무작위 위치에 무작위 크기의 cutout 적용 (특정 주파수 성분 제거)
4. 역 FFT를 적용하여 이미지 공간으로 변환
5. 변환된 채널을 합쳐 새로운 RGB 이미지 생성

## 활용 방법

1. 설치 요구사항:
   ```
   pip install numpy opencv-python
   ```

2. 명령줄에서 실행:
   ```
   python fft_cutout_augmentation.py --input path/to/image.jpg --output ./output_folder --count 50
   ```

3. 매개변수:
   - `--input`: 입력 이미지 경로 (필수)
   - `--output`: 출력 이미지가 저장될 디렉토리 (기본값: ./filtered_images)
   - `--count`: 생성할 증강 이미지의 수 (기본값: 100)

## 사용 예시

```python
# 모듈로 가져와서 사용하기
from fft_cutout_augmentation import apply_fft_cutout

# 이미지 경로와 출력 디렉토리 지정
apply_fft_cutout('input.jpg', './augmented_images', num_images=50)
```

## 응용 분야

- 딥러닝 모델 훈련을 위한 데이터 증강
- 이미지 분류 및 객체 탐지 성능 향상
- 주파수 도메인 특성에 강건한 모델 훈련
- 적대적 샘플 생성 및 방어 기법 연구

## 참고 사항

주파수 도메인에서의 augmentation은 공간 도메인의 augmentation과 다른 특성을 가집니다. 특히 저주파 영역을 제거하면 이미지의 전체적인 구조가 변경되고, 고주파 영역을 제거하면 세부 디테일이 손실됩니다. 이러한 특성을 활용하여 다양한 데이터 증강 효과를 얻을 수 있습니다.