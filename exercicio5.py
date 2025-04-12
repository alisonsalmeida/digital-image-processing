import cv2
import numpy as np
import matplotlib.pyplot as plt

# imagem com ruido
noisy_img = cv2.imread('images/BARBARA02.png')

# Aplicar o filtro de mediana para remover o ruído
filtered_img = cv2.medianBlur(noisy_img, 5)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(noisy_img, cmap='gray')
axs[0].set_title("Imagem com Ruído")
axs[0].axis('off')

axs[1].imshow(filtered_img, cmap='gray')
axs[1].set_title("Imagem Filtrada")
axs[1].axis('off')

plt.tight_layout()
plt.show()
