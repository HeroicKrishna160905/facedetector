import cv2
import numpy as np
import tensorflow as tf

# 
model = tf.keras.models.load_model('face_detector_model.keras')

IMAGE_SIZE = (120, 120)  

def preprocess(frame):
    img = cv2.resize(frame, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --- Start the webcam ---
cap = cv2.VideoCapture(0)  # 

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    input_img = preprocess(frame)
    pred_class, pred_bbox = model.predict(input_img)
    print("pred_class:", pred_class, "pred_bbox:", pred_bbox)  # Debug print

    pred_class = pred_class[0][0]
    pred_bbox = pred_bbox[0]

    if pred_class > 0.5:  
        h, w, _ = frame.shape
        x, y, bw, bh = pred_bbox
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            frame,
            f"Face: {pred_class:.2f}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    cv2.imshow("Live Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
