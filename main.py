import cv2

from src.detection.yolo_detector import YOLODetector
from src.visualization.draw import draw_detections


def run_pipeline(video_path):

    cap = cv2.VideoCapture(video_path)

    detector = YOLODetector()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        detections = detector.detect(frame)

        frame = draw_detections(frame, detections)

        cv2.imshow("PRAS Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    video_path = "data/raw_videos/test.mp4"

    run_pipeline(video_path)