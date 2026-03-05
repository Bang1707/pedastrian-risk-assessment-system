from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

        self.PERSON_CLASS = 0 #person class id
        self.CAR_CLASS = 2 #car class id 

    def detect(self, frame):

        results = self.model(frame)

        detections = []

        for r in results:
            boxes = r.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                if cls_id in [self.PERSON_CLASS, self.CAR_CLASS]:

                    detections.append({
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })

        return detections