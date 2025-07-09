import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

window_name = "Cámara"
cv2.namedWindow(window_name)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error leyendo frame.")
        break

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
