import cv2
import matplotlib.pyplot as plt
import numpy as np


def otsu_segment(img: np.ndarray) -> np.ndarray:
    # Aplicar o algoritmo de Otsu
    _, otsu_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_result

img_plate = cv2.imread('images/placa-02.jpg')
otsu_img_plate = otsu_segment(cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY))

img_boat = cv2.imread('images/barco.jpg')
otsu_img_boat = otsu_segment(cv2.cvtColor(img_boat, cv2.COLOR_BGR2GRAY))

_, axs = plt.subplots(2, 2, figsize=(16, 5))
axs = axs.ravel()
axs[0].imshow(cv2.cvtColor(img_plate, cv2.COLOR_BGR2RGB))
axs[0].set_title("Placa")
axs[0].axis('off')

axs[1].imshow(otsu_img_plate, cmap='gray')
axs[1].set_title("Otsu Placa")
axs[1].axis('off')

axs[2].imshow(cv2.cvtColor(img_boat, cv2.COLOR_BGR2RGB))
axs[2].set_title("Barco")
axs[2].axis('off')

axs[3].imshow(otsu_img_boat, cmap='gray')
axs[3].set_title("Otsu Barco")
axs[3].axis('off')

plt.tight_layout()
plt.show()
