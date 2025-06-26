import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

img = cv2.imread("images/woman.tif", cv2.IMREAD_GRAYSCALE)
img_filtrada = img.copy()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

img_plot1 = axs[0].imshow(img, cmap='gray')
axs[0].set_title("Original")
axs[0].axis('off')

img_plot2 = axs[1].imshow(img_filtrada, cmap='gray')
axs[1].set_title("Imagem Filtrada")
axs[1].axis('off')

ax_filter = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_region = plt.axes([0.2, 0.05, 0.65, 0.03])

slider_filter = Slider(ax_filter, 'Filtro', 3, 51, valinit=21, valstep=2)
slider_region = Slider(ax_region, 'Região', 10, 150, valinit=50, valstep=2)

def on_click(event, image):
    if event.inaxes != axs[1]:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    raio = int(slider_filter.val)
    regiao = int(slider_region.val)

    x1 = max(0, x - regiao // 2)
    y1 = max(0, y - regiao // 2)
    x2 = min(img.shape[1], x + regiao // 2)
    y2 = min(img.shape[0], y + regiao // 2)

    roi = img[y1:y2, x1:x2]
    h, w = roi.shape

    dft = np.fft.fft2(roi)
    dft_shift = np.fft.fftshift(dft)

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    cx, cy = w // 2, h // 2
    gauss = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * (raio**2)))

    dft_filtrado = dft_shift * gauss

    dft_ishift = np.fft.ifftshift(dft_filtrado)
    img_suavizada = np.fft.ifft2(dft_ishift)
    img_suavizada = np.abs(img_suavizada).astype(np.uint8)

    image[y1:y2, x1:x2] = img_suavizada

    img_plot2.set_data(image)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, img_filtrada))
plt.show()
