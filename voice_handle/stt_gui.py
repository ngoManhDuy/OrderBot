#!/usr/bin/env python3
# filepath: /home/ngonanhduy/Documents/asr/final_project/stt_gui.py

import sys
import threading
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QTextEdit, QLabel, QFrame, 
                             QScrollArea, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

from stt_whisper import STT_module


class STTWorker(QThread):
    """Worker thread for STT processing to avoid blocking the GUI"""
    
    # Signals to communicate with the main GUI thread
    status_update = pyqtSignal(str)
    transcription_ready = pyqtSignal(str, str)  # raw_text, corrected_text
    error_occurred = pyqtSignal(str)
    listening_status = pyqtSignal(bool)  # True when listening, False when processing
    
    def __init__(self, model_name="openai/whisper-medium"):
        super().__init__()
        self.stt_module = None
        self.model_name = model_name
        self.should_stop = False
        self.is_running = False
        
    def run(self):
        """Main worker thread function"""
        try:
            # Initialize STT module with status callback
            def status_callback(message):
                self.status_update.emit(message)
            
            self.status_update.emit("Initializing STT module...")
            self.stt_module = STT_module(model_name=self.model_name, status_callback=status_callback)
            self.status_update.emit("STT module loaded successfully!")
            
            self.is_running = True
            
            while not self.should_stop:
                self.listening_status.emit(True)
                
                # Process single recording
                raw_text, corrected_text = self.stt_module.process_single_recording(max_duration=30)
                
                if self.should_stop:
                    break
                    
                if raw_text is None:
                    self.status_update.emit("No speech detected, continuing to listen...")
                    time.sleep(0.5)
                    continue
                
                self.listening_status.emit(False)
                
                # Emit the results
                self.transcription_ready.emit(raw_text, corrected_text)
                
                # Brief pause before next iteration
                time.sleep(1)
                
        except Exception as e:
            self.error_occurred.emit(f"STT Error: {str(e)}")
        finally:
            self.is_running = False
            self.listening_status.emit(False)
    
    def stop(self):
        """Stop the worker thread"""
        self.should_stop = True
        self.status_update.emit("Stopping STT system...")


class TranscriptionDisplay(QWidget):
    """Widget to display individual transcription results"""
    
    def __init__(self, timestamp, raw_text, corrected_text):
        super().__init__()
        self.setup_ui(timestamp, raw_text, corrected_text)
    
    def setup_ui(self, timestamp, raw_text, corrected_text):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Timestamp
        time_label = QLabel(f"🕐 {timestamp}")
        time_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(time_label)
        
        # Raw transcription
        raw_label = QLabel("Raw Transcription:")
        raw_label.setStyleSheet("font-weight: bold; color: #888; font-size: 13px;")
        layout.addWidget(raw_label)
        
        raw_text_label = QLabel(raw_text)
        raw_text_label.setWordWrap(True)
        raw_text_label.setStyleSheet("color: #666; font-size: 14px; margin-left: 10px; margin-bottom: 5px;")
        layout.addWidget(raw_text_label)
        
        # Corrected transcription
        corrected_label = QLabel("LLM Corrected:")
        corrected_label.setStyleSheet("font-weight: bold; color: #2c5282; font-size: 13px;")
        layout.addWidget(corrected_label)
        
        corrected_text_label = QLabel(corrected_text)
        corrected_text_label.setWordWrap(True)
        corrected_text_label.setStyleSheet("color: #2d3748; font-size: 15px; font-weight: 500; margin-left: 10px;")
        layout.addWidget(corrected_text_label)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #e2e8f0;")
        layout.addWidget(separator)
        
        self.setLayout(layout)


class STTMainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.transcription_count = 0
        self.setup_ui()
        self.setup_worker()
        
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Coffee Shop STT System")
        self.setGeometry(100, 100, 800, 600)
        
        # Apply modern styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f7fafc;
            }
            QPushButton {
                background-color: #4299e1;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3182ce;
            }
            QPushButton:pressed {
                background-color: #2c5282;
            }
            QPushButton:disabled {
                background-color: #a0aec0;
            }
            QLabel {
                color: #2d3748;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Testing Voice Recognition System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c5282; margin-bottom: 10px;")
        main_layout.addWidget(title)
        
        # Status section
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Box)
        status_frame.setStyleSheet("QFrame { background-color: white; border: 1px solid #e2e8f0; border-radius: 8px; }")
        status_layout = QVBoxLayout(status_frame)
        
        # Status label
        self.status_label = QLabel("Ready to start...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; color: #4a5568; padding: 10px;")
        status_layout.addWidget(self.status_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Listening")
        self.start_button.clicked.connect(self.start_listening)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_listening)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #e53e3e; } QPushButton:hover { background-color: #c53030; }")
        
        self.clear_button = QPushButton("Clear Results")
        self.clear_button.clicked.connect(self.clear_results)
        self.clear_button.setStyleSheet("QPushButton { background-color: #38a169; } QPushButton:hover { background-color: #2f855a; }")
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.clear_button)
        status_layout.addLayout(button_layout)
        
        main_layout.addWidget(status_frame)
        
        # Results section
        results_label = QLabel("Transcription Results:")
        results_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c5282; margin-top: 10px;")
        main_layout.addWidget(results_label)
        
        # Scrollable results area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: 1px solid #e2e8f0; border-radius: 8px; background-color: white; }")
        
        # Widget to contain all transcription results
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.results_layout.setSpacing(5)
        
        self.scroll_area.setWidget(self.results_widget)
        main_layout.addWidget(self.scroll_area)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def setup_worker(self):
        """Setup the STT worker thread"""
        self.worker = STTWorker()
        self.worker.status_update.connect(self.update_status)
        self.worker.transcription_ready.connect(self.add_transcription)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.listening_status.connect(self.update_listening_status)
        
    def start_listening(self):
        """Start the STT system"""
        if self.worker and not self.worker.is_running:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.worker.should_stop = False
            self.worker.start()
            self.statusBar().showMessage("STT System Running")
            
    def stop_listening(self):
        """Stop the STT system"""
        if self.worker and self.worker.is_running:
            self.worker.stop()
            self.worker.wait()  # Wait for thread to finish
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.statusBar().showMessage("STT System Stopped")
            
    def clear_results(self):
        """Clear all transcription results"""
        # Remove all child widgets from results layout
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.transcription_count = 0
        self.statusBar().showMessage("Results cleared")
        
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
        
    def update_listening_status(self, is_listening):
        """Update UI based on listening status"""
        if is_listening:
            self.status_label.setStyleSheet("font-size: 16px; color: #38a169; padding: 10px; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("font-size: 16px; color: #e53e3e; padding: 10px; font-weight: bold;")
            
    def add_transcription(self, raw_text, corrected_text):
        """Add a new transcription result to the display"""
        self.transcription_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Create transcription display widget
        transcription_widget = TranscriptionDisplay(timestamp, raw_text, corrected_text)
        self.results_layout.insertWidget(0, transcription_widget)  # Add to top
        
        # Auto-scroll to top to show latest result
        self.scroll_area.verticalScrollBar().setValue(0)
        
        # Update status
        self.statusBar().showMessage(f"Total transcriptions: {self.transcription_count}")
        
    def handle_error(self, error_message):
        """Handle errors from the worker thread"""
        QMessageBox.critical(self, "STT Error", error_message)
        self.stop_listening()
        
    def closeEvent(self, event):
        """Handle application close event"""
        if self.worker and self.worker.is_running:
            self.worker.stop()
            self.worker.wait()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Coffee Shop STT System")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = STTMainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
