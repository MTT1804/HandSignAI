import sys, cv2
# Program do testowania czy kamera o określonym indeksie działa poprawnie
# i wyświetla obraz w czasie rzeczywistym.
# Indeksowanie kamer w systemie zaczyna się od 0
# q - zakończenie programu
if len(sys.argv) != 2:
    print("Podaj indeks kamery jako argument wywołania programu.")
    sys.exit(1)

try:
    camera_id = int(sys.argv[1])
except ValueError:
    print("Podaj poprawny indeks kamery jako argument wywołania programu.")
    sys.exit(1)
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    print("Kamera o indeksie", camera_id, "nie działa.")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Obraz z kamery", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
