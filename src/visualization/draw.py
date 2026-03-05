import cv2


def draw_detections(frame, detections):

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]
        cls = det["class_id"]
        conf = det["confidence"]

        if cls == 0:
            label = "Person"
            color = (0, 255, 0)

        elif cls == 2:
            label = "Car"
            color = (255, 0, 0)

        else:
            label = "Object"
            color = (200, 200, 200)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return frame