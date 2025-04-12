import cv2
import matplotlib.pyplot as plt

# Carregar a imagem original
img = cv2.imread('images/paisagem.jpg')

# Converter a imagem para escala de cinza
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
mask = gray_img - blurred_img

# Filtro high-boost (K = 1.0)
K = 0.5
high_boost_img = gray_img + (K * mask)

# Exibir a imagem original, a imagem borrada e a imagem high-boost
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Imagem original
axs[0].imshow(gray_img, cmap='gray')
axs[0].set_title("Imagem Original")
axs[0].axis('off')

# Imagem borrada
axs[1].imshow(blurred_img, cmap='gray')
axs[1].set_title("Imagem Borrada")
axs[1].axis('off')

# Imagem com filtro high-boost
axs[2].imshow(high_boost_img, cmap='gray')
axs[2].set_title("Imagem High-Boost")
axs[2].axis('off')

plt.tight_layout()
plt.show()
