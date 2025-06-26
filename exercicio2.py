import cv2
import matplotlib.pyplot as plt


img1 = cv2.imread('images/imagem-01.tif')
img2 = cv2.imread('images/imagem-02.tif')

sad = cv2.absdiff(img1, img2)

_, binary = cv2.threshold(sad, 80, 255, cv2.THRESH_BINARY)

fig, axs = plt.subplots(1, 4, figsize=(16, 5))

axs[0].imshow(img1, cmap='gray')
axs[0].set_title("Imagem 01")
axs[0].axis('off')

axs[1].imshow(img2, cmap='gray')
axs[1].set_title("Imagem 02")
axs[1].axis('off')

axs[2].imshow(sad, cmap='gray')
axs[2].set_title("Imagem Diferença (SAD)")
axs[2].axis('off')

axs[3].imshow(binary, cmap='gray')
axs[3].set_title("Diferenças em Branco")
axs[3].axis('off')

plt.tight_layout()
plt.show()
