import sys
import cv2
import numpy as np
import yaml
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QSlider, QProgressBar, QStyle, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor


from detection.yolo_detector import YOLOTinyDetector
from classification.classifier import Classifier

class DetectionSmoother:
    def __init__(self, smoothing_frames=5):
        self.smoothing_frames = smoothing_frames
        self.detection_history = []

    def update(self, current_detections):
        self.detection_history.append(current_detections)
        if len(self.detection_history) > self.smoothing_frames:
            self.detection_history.pop(0)

    def get_smoothed_detections(self):
        if not self.detection_history:
            return []

        smoothed_detections = []
        all_detections = [det for frame_dets in self.detection_history for det in frame_dets]

        for det in all_detections:
            similar_dets = [d for d in all_detections if self.iou(det['bbox'], d['bbox']) > 0.3]
            if len(similar_dets) >= 2: 
                avg_bbox = self.average_bbox([d['bbox'] for d in similar_dets])
                avg_conf = sum(d['confidence'] for d in similar_dets) / len(similar_dets)
                smoothed_detections.append({
                    'bbox': avg_bbox,
                    'class': det['class'],
                    'confidence': avg_conf
                })

        return smoothed_detections

    @staticmethod
    def average_bbox(bboxes):
        # Calculate average coordinates of bounding box
        avg_bbox = [
            sum(box[i] for box in bboxes) / len(bboxes)
            for i in range(4)
        ]
        return [int(coord) for coord in avg_bbox]

    @staticmethod
    def iou(box1, box2):
        # Calculate Intersection over Union (IoU) between two bounding box
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

class ProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    frame_processed_signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    alert_signal = pyqtSignal(str)

    def __init__(self, config, video_path, config_path):
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.config_path = config_path
        self.detector = YOLOTinyDetector(self.config)
        self.classifier = Classifier(self.config_path)
        self.detection_smoother = DetectionSmoother()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = []

        for i in range(0, total_frames, self.config['performance']['frame_skip']):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, tuple(self.config['performance']['target_resolution']))

            #detections = self.detector.detect(frame)
            #processed_frame = self.process_frame(frame, detections)
            processed_frame = self.process_frame(frame)
            processed_frames.append(processed_frame)

            self.frame_processed_signal.emit(processed_frame)
            self.progress_signal.emit(int((i + 1) / total_frames * 100))

            if self.isInterruptionRequested():
                break

        cap.release()
        self.finished_signal.emit(processed_frames)

    def process_frame(self, frame):
        detections = self.detector.detect(frame)
        self.detection_smoother.update(detections)
        smoothed_detections = self.detection_smoother.get_smoothed_detections()

        for detection in smoothed_detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            object_img = frame[y1:y2, x1:x2]
            
            try:
                classification_result = self.classifier.classify(object_img)
                if classification_result:
                    color = (0, 255, 0)  
                    label = f"Animal ({classification_result['confidence']:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    self.alert_signal.emit(self.config['alerts']['animal_detected'])
                    print(f"Animal detected and classified: {label}")
            except Exception as e:
                print(f"Error in classification: {str(e)}")

        return frame


class PlaybackThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal()

    def __init__(self, processed_frames):
        super().__init__()
        self.processed_frames = processed_frames
        self.current_frame = 0
        self._run_flag = True
        self.delay = 0

    def run(self):
        while self._run_flag and self.current_frame < len(self.processed_frames):
            self.change_pixmap_signal.emit(self.processed_frames[self.current_frame])
            self.current_frame += 1
            if self.delay > 0:
                self.msleep(self.delay)
        self.finished_signal.emit()

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_delay(self, delay):
        self.delay = delay

class MainWindow(QMainWindow):
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path  
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.setWindowTitle(self.config['gui']['window_title'])
        self.setGeometry(100, 100, self.config['gui']['window_size']['width'], self.config['gui']['window_size']['height'])
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLabel {
                font-size: 14px;
            }
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Video display area
        self.video_frame = QFrame()
        self.video_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.video_frame.setLineWidth(2)
        self.video_layout = QVBoxLayout(self.video_frame)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_layout.addWidget(self.video_label)
        self.main_layout.addWidget(self.video_frame, 3)

        # Control buttons
        self.button_layout = QHBoxLayout()
        self.load_button = QPushButton("Upload Video")
        self.load_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.process_button = QPushButton("Process Video")
        self.process_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.process_button.setEnabled(False)
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_button.setEnabled(False)
        
        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.process_button)
        self.button_layout.addWidget(self.play_pause_button)
        self.main_layout.addLayout(self.button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.main_layout.addWidget(self.progress_bar)

        # Speed control
        self.speed_layout = QHBoxLayout()
        self.speed_label = QLabel("Playback Speed:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(50)
        self.speed_layout.addWidget(self.speed_label)
        self.speed_layout.addWidget(self.speed_slider, 1)
        self.main_layout.addLayout(self.speed_layout)

        # Alert label
        self.alert_label = QLabel()
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setStyleSheet("""
            background-color: #f44336;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
        """)
        self.alert_label.hide()
        self.main_layout.addWidget(self.alert_label)

        # Connect signals
        self.load_button.clicked.connect(self.load_video)
        self.process_button.clicked.connect(self.process_video)
        self.play_pause_button.clicked.connect(self.play_pause_video)
        self.speed_slider.valueChanged.connect(self.change_speed)

        # Initialize variables
        self.video_path = None
        self.processed_frames = None
        self.processing_thread = None
        self.playback_thread = None
        self.video_playing = False

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.video_path = file_name
            self.process_button.setEnabled(True)
            self.show_alert("Video uploaded. Press 'Process Video' to start processing.")

    def process_video(self):
        if self.video_path:
            self.processing_thread = ProcessingThread(self.config, self.video_path, self.config_path)
            self.processing_thread.progress_signal.connect(self.update_progress)
            self.processing_thread.frame_processed_signal.connect(self.update_image)
            self.processing_thread.finished_signal.connect(self.processing_finished)
            self.processing_thread.error_signal.connect(self.show_error)
            self.processing_thread.alert_signal.connect(self.show_alert) 
            self.processing_thread.start()
            self.process_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.show_alert("Processing...")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if value % 10 == 0: 
            self.show_alert(f"Processing... {value}% completed")

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def processing_finished(self, processed_frames):
        self.processed_frames = processed_frames
        self.play_pause_button.setEnabled(True)
        self.show_alert("Processing completed. Press 'Play' to start playback.")

    def play_pause_video(self):
        if not self.video_playing:
            if self.processed_frames:
                if not self.playback_thread:
                    self.playback_thread = PlaybackThread(self.processed_frames)
                    self.playback_thread.change_pixmap_signal.connect(self.update_image)
                    self.playback_thread.finished_signal.connect(self.playback_finished)
                self.playback_thread.start()
                self.video_playing = True
                self.play_pause_button.setText("Pause")
        else:
            if self.playback_thread:
                self.playback_thread.stop()
            self.video_playing = False
            self.play_pause_button.setText("Play")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def show_alert(self, message):
        self.alert_label.setText(message)
        self.alert_label.show()
        QTimer.singleShot(3000, self.alert_label.hide)

    def change_speed(self, value):
        speed = value / 100.0  
        if self.playback_thread:
            self.playback_thread.set_delay(int(100 / speed))
        self.speed_label.setText(f"Playback Speed: {speed:.2f}x")

    def show_error(self, error_message):
        print(f"Error: {error_message}")
        self.show_alert(f"Error: {error_message}")

    def playback_finished(self):
        self.video_playing = False
        self.play_pause_button.setText("Play")
        self.show_alert("Reproduction completed.")

    def closeEvent(self, event):
        if self.processing_thread:
            self.processing_thread.wait()
        if self.playback_thread:
            self.playback_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow('config/config.yaml')
    main_window.show()
    sys.exit(app.exec_())