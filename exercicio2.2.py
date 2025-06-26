import cv2
import numpy as np
import matplotlib.pyplot as plt


def high_frequency_enhancement(img_gray, k=1.5, radius=30):
    # pega o tamnho da imagem
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2

    # Aplica DFT e centraliza o espectro
    dft = np.fft.fft2(img_gray)
    dft_shift = np.fft.fftshift(dft)

    # cria a máscara passa-altas (Hhp): zero no centro, 1 fora
    mask = np.ones((rows, cols), np.float32)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    # aplicando filtro: [1 + k * Hhp(u,v)] * F(u,v)
    filtered_dft = (1 + k * mask) * dft_shift

    # faz DFT inversa para pegar a imagem realçada
    f_ishift = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # normaliza para exibir
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back

ultrasound = cv2.imread("images/ultrasound_triplets.png", cv2.IMREAD_GRAYSCALE)
ultrasound_enhancement = high_frequency_enhancement(ultrasound, k=1.5, radius=40)

radiograph = cv2.imread("images/radiograph.png", cv2.IMREAD_GRAYSCALE)
radiograph_enhancement = high_frequency_enhancement(radiograph, k=1.8, radius=50)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(ultrasound, cmap='gray')
plt.title("Ultrasound - Original")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(ultrasound_enhancement, cmap='gray')
plt.title("Ultrasound - Realçada")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(radiograph, cmap='gray')
plt.title("Radiograph - Original")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(radiograph_enhancement, cmap='gray')
plt.title("Radiograph - Realçada")
plt.axis('off')

plt.tight_layout()
plt.show()