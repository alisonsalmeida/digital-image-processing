import cv2


cena1_img = cv2.imread('images/cena1.png')
cena2_img = cv2.imread('images/cena2.png')

sad = cv2.absdiff(cena1_img, cena2_img)

_, binary = cv2.threshold(sad, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Diferenca (SAD)", sad)
cv2.imshow("Binarizada", binary)

cv2.waitKey(0)
cv2.destroyAllWindows()
