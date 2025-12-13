import cv2
import numpy as np
import matplotlib.pyplot as plt

def homomorphic_filtering_improved(img_color, cutoff=60, gamma_low=0.3, gamma_high=1.5, c=1):

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    v_float = v.astype(np.float64) / 255.0
    img_log = np.log1p(v_float)
    img_fft_shift = np.fft.fftshift(np.fft.fft2(img_log))
    
    rows, cols = v.shape
    center_row, center_col = rows // 2, cols // 2

    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    D_sq = (xx - center_col)**2 + (yy - center_row)**2
    
    D0_sq = cutoff**2
    gaussian_term = 1 - np.exp(-c * D_sq / (D0_sq + 1e-6))
    H = (gamma_high - gamma_low) * gaussian_term + gamma_low
            
    img_fft_filtered = img_fft_shift * H
    img_ifft_shift = np.fft.ifftshift(img_fft_filtered)
    img_ifft = np.fft.ifft2(img_ifft_shift)
    
    v_filtered = np.real(img_ifft)
    v_final = np.expm1(v_filtered) 

    v_final_norm = cv2.normalize(v_final, None, 0, 255, cv2.NORM_MINMAX)
    v_final_uint8 = v_final_norm.astype(np.uint8)

    img_hsv_filtered = cv2.merge([h, s, v_final_uint8])

    img_bgr_filtered = cv2.cvtColor(img_hsv_filtered, cv2.COLOR_HSV2BGR)
    
    return img_bgr_filtered

img_orig_color = cv2.imread('../../dataset/uneven_illumination/dataset_classroom.png')

img_homomorphic_color = homomorphic_filtering_improved(
    img_orig_color, 
    cutoff=1,       
    gamma_low=0.005,  
    gamma_high=4
)

# 3. 결과 시각화
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.title('Original Color Image')
plt.imshow(cv2.cvtColor(img_orig_color, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Improved Homomorphic Filtering Result')
plt.imshow(cv2.cvtColor(img_homomorphic_color, cv2.COLOR_BGR2RGB))

plt.axis('off')

plt.show()

# 파일로 저장
cv2.imwrite('../../result/uneven_illumination/homomorphic.jpg', img_homomorphic_color)