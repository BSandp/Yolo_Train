import cv2

video_path = "video/video_yolo_640x640.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el video")
    exit()

# Obtener FPS real
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

delay = int(1000 / fps)

print(f"FPS del video: {fps}")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Fin del video")
        break

    # Mostrar video sin procesamiento
    cv2.imshow("Video original", frame)

    if cv2.waitKey(delay) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()