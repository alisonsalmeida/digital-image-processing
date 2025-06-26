import cv2
import numpy as np
import time

# Carregar imagem de fundo e redimensionar
background = cv2.imread('images/background.png')

# Inicializar a captura de vídeo (0 = webcam padrão)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

# Obter tamanho dos frames da câmera
ret, frame = cap.read()
if not ret:
    print("Erro ao capturar o primeiro frame.")
    exit()

height, width = frame.shape[:2]
background = cv2.resize(background, (width, height))

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_low = (np.array([0, 120, 70]), np.array([170, 120, 70]))
    red_high  = (np.array([10, 255, 255]), np.array([180, 255, 255]))

    mask1 = cv2.inRange(hsv, red_low[0], red_high[0])
    mask2 = cv2.inRange(hsv, red_low[1], red_high[1])
    red_mask = cv2.bitwise_or(mask1, mask2)

    inverted_mask = cv2.bitwise_not(red_mask)
    red_objects = cv2.bitwise_and(frame, frame, mask=red_mask)

    background_mask = cv2.bitwise_and(background, background, mask=inverted_mask)

    result = cv2.add(red_objects, background_mask)

    cv2.imshow("Video", result)
    time.sleep(1/24)

    # Sair com tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerrar
cap.release()
cv2.destroyAllWindows()
