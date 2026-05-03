from ultralytics import YOLO
import cv2

# Cargar modelo
model = YOLO("best.pt")

video_path = "video/video_yolo_640x640.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el video")
    exit()

# 🔥 Obtener FPS real del video
fps = cap.get(cv2.CAP_PROP_FPS)

# Evitar división por 0
if fps == 0:
    fps = 25

delay = int(1000 / fps)

print(f"FPS del video: {fps}")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Fin del video")
        break

    # Inferencia
    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Inference", annotated_frame)

    # 🔥 Usar delay dinámico
    if cv2.waitKey(delay) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()