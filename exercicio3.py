import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/coluna.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_negativa = 255 - img_rgb

# Exibir as imagens lado a lado
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(img_rgb)
axs[0].set_title("Imagem Original")
axs[0].axis('off')

axs[1].imshow(img_negativa)
axs[1].set_title("Imagem Negativa")
axs[1].axis('off')

plt.tight_layout()
plt.show()
