import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/paisagem.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

R = img[:, :, 0].astype(float)
G = img[:, :, 1].astype(float)
B = img[:, :, 2].astype(float)

Y1 = 0.299 * R + 0.587 * G + 0.114 * B
gray_scale = Y1.astype(np.uint8)

# Mostrar imagens lado a lado
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(img)
axs[0].set_title("Imagem Original")
axs[0].axis('off')

axs[1].imshow(gray_scale, cmap='gray')
axs[1].set_title("Imagem em Tons de Cinza")
axs[1].axis('off')

plt.tight_layout()
plt.show()
