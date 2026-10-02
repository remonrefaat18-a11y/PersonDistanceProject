from ultralytics import YOLO
import cv2


KNOWN_HEIGHTS = {
    "bottle": 23.0,
    "cell phone": 16.0,

}

FOCAL = 487.5

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        name = model.names[int(box.cls[0])]
        if name not in KNOWN_HEIGHTS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pixel_h = y2 - y1
        if pixel_h <= 0:
            continue

        dist = (KNOWN_HEIGHTS[name] * FOCAL) / pixel_h
        cv2.putText(annotated_frame, f"{dist:.1f} cm", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

       
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            CALIB_DIST = 50.0
            FOCAL = (pixel_h * CALIB_DIST) / KNOWN_HEIGHTS[name]
            print("New FOCAL =", FOCAL)

    cv2.imshow("YOLO Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()