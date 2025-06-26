import cv2
import numpy as np
import matplotlib.pyplot as plt

def butterworth_lowpass(shape, cutoff, order):
    rows, cols = shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x - cols // 2, y - rows // 2)
    D = np.sqrt(X**2 + Y**2)
    H = 1 / (1 + (D / cutoff)**(2 * order))
    
    return H

img = cv2.imread("images/dedo.tif", cv2.IMREAD_GRAYSCALE)

dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

cutoff = 200    # frequência de corte
order = 2      # ordem do filtro
butter = butterworth_lowpass(img.shape, cutoff, order)

filtro_aplicado = dft_shift * butter

f_ishift = np.fft.ifftshift(filtro_aplicado)
img_filtrada = np.fft.ifft2(f_ishift)
img_filtrada = np.abs(img_filtrada)

img_filtrada = cv2.normalize(img_filtrada, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

_, img_bin = cv2.threshold(img_filtrada, 127, 255, cv2.THRESH_OTSU)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Imagem Original")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_filtrada, cmap='gray')
plt.title("Filtrada (Butterworth)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_bin, cmap='gray')
plt.title("Binarizada")
plt.axis('off')

plt.tight_layout()
plt.show()
