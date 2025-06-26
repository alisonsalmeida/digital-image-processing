import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/hubble.png", cv2.IMREAD_GRAYSCALE)
_, bin_img = cv2.threshold(img, 200, 255, cv2.THRESH_OTSU)
contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

points = [(57, 315), (101, 242), (266, 215), (175, 325), (466, 234)]
raio_max = 10

# Imagem de saída (preta)
saida = np.zeros_like(img)

# Função auxiliar: calcular distância euclidiana
def distancia(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    for px, py in points:
        if distancia((cx, cy), (px, py)) <= raio_max:
            cv2.drawContours(saida, [cnt], -1, 255, -1)
            break

# Mostrar resultado
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Imagem Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(saida, cmap='gray')
plt.title("5 Pontos Selecionados por Posição")
plt.axis('off')

plt.tight_layout()
plt.show()
