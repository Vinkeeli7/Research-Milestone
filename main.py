from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QTableWidget,QTableWidgetItem,QMessageBox,QScrollArea
from PySide6.QtCore import Qt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.linalg import lstsq
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Regression Model")
        self.data = 0
        self.model = None
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # CSV button
        self.csv_button = QPushButton("Open CSV dataset")
        self.csv_button.clicked.connect(self.load_csv)
        layout.addWidget(self.csv_button)

        # Dataset Informaiton
        self.info_label = QLabel("No dataset loaded.")
        layout.addWidget(self.info_label)

        # Input feature selection
        self.input_label = QLabel("Select Input Features (One or more X):")
        layout.addWidget(self.input_label)
        # create scroll area for input features
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMaximumHeight(150)
        # checkboxes for input features and add them to the scroll area
        self.input_widget = QWidget()
        self.input_layout = QVBoxLayout(self.input_widget)
        self.scroll_area.setWidget(self.input_widget)
        layout.addWidget(self.scroll_area)
        self.input_checkboxes = []

        # Target feature selection
        self.target_label = QLabel("Select Target Feature (Y):")
        layout.addWidget(self.target_label)
        # one checkbox for target feature
        self.target_widget = QWidget()
        self.target_layout = QVBoxLayout(self.target_widget)
        layout.addWidget(self.target_widget)
        self.target_checkbox = None

    def load_csv(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                # Only have numerical data
                self.numeric_data = self.data.select_dtypes(include=[np.number])
                # checks for columns
                if len(self.numeric_data.columns) < 2:
                    QMessageBox.warning(self, "Warning", "The dataset must have at least two columns for simple linear regression.")
                    return
                self.info_label.setText(f"Dataset Information:\nRows: {self.data.shape[0]}, Columns: {self.data.shape[1]}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV file:\n{str(e)}")
app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec())