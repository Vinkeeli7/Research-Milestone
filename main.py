from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QTableWidget,QTableWidgetItem,QMessageBox,QScrollArea, QComboBox
from PySide6.QtCore import Qt
from matplotlib.backend_bases import FigureCanvasBase
import pandas as pd
import numpy as np
import sys
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
        self.target_box = QComboBox()
        self.target_box.setPlaceholderText("Select Target Feature")
        layout.addWidget(self.target_box)
        self.target_checkbox = None
        # Train button
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)
        # Results
        self.result_label = QLabel("Load a dataset and select features to train the model.")
        layout.addWidget(self.result_label)
    def load_csv(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                # Only have numerical data
                self.numeric_data = self.data.select_dtypes(include=[np.number])
                columns = self.numeric_data.columns.tolist()
                # checks for columns
                if len(columns) < 2:
                    QMessageBox.warning(self, "Warning", "The dataset must have at least two columns for simple linear regression.")
                    return
                self.info_label.setText(f"Dataset Information:\nRows: {self.data.shape[0]}, Columns: {self.data.shape[1]}")
                for checkbox in self.input_checkboxes:
                    self.input_layout.removeWidget(checkbox)
                    checkbox.deleteLater()
                self.input_checkboxes = []
                # create checkboxes for input features
                for column in columns:
                    checkbox = QPushButton(column)
                    checkbox.setCheckable(True)
                    self.input_layout.addWidget(checkbox)
                    self.input_checkboxes.append(checkbox)
                # clear previous and create checkbox for target feature
                self.target_box.clear()
                self.target_box.addItems(columns)
                self.target_box.setCurrentIndex(1) # set default target feature to the second column
                self.result_label.setText("Select input features and target feature, then click 'Train Model' to train the linear regression model.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV file:\n{str(e)}")
    def train_model(self):
        if self.data is None or self.data.empty:
            QMessageBox.warning(self, "Warning", "No dataset loaded. Please load a CSV file first.")
            return
        selected_inputs = [checkbox.text() for checkbox in self.input_checkboxes if checkbox.isChecked()]
        target_feature = self.target_box.currentText()
        if not selected_inputs:
            QMessageBox.warning(self, "Warning", "Please select at least one input feature (X).")
            return
        if target_feature in selected_inputs:
            QMessageBox.warning(self, "Warning", "Target feature (Y) cannot be one of the input features (X).")
            return
        model_data = self.numeric_data[selected_inputs + [target_feature]].copy()
        model_data = model_data.dropna()
        if model_data.empty:
            QMessageBox.warning(self, "Warning", "The selected features contain only NaN values. Please select different features.")
            return
        X = model_data[selected_inputs].to_numpy(dtype=float)
        y = model_data[target_feature].to_numpy(dtype=float)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # calculate regression coefficients using least squares method
        coefficients,residuals,rank,s  = np.linalg.lstsq(X_b, y,rcond=None)
        intercept = coefficients[0]
        slopes = coefficients[1:]
        # calculate predictions
        y_pred = X_b @ coefficients
        # calculate R-squared
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        # check if ss_total is zero to avoid division by zero
        if ss_total == 0:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_residual / ss_total)
        self.coefficients = coefficients
        self.input_features = selected_inputs
        # build regression equation string
        equation = f"{target_feature} = {intercept:.4f}"
        for i, feature in enumerate(selected_inputs):
            equation += f" + ({slopes[i]:.4f} * {feature})"
        # format coefficients for display
        coefficient_text = "\n".join([f"{feature}: {slopes[i]:.4f}"
                for i, feature in enumerate(selected_inputs)])
        # display results
        self.result_label.setText("Model trained successfully!\n\n" 
                    f"Input features: {', '.join(selected_inputs)}\n"
                    f"Target feature: {target_feature}\n\n"
                    f"Regression Equation:\n{equation}\n\n"
                    f"Intercept: {intercept:.4f}\n"
                    f"Coefficients:\n{coefficient_text}\n\n" 
                    f"R-squared: {r_squared:.4f}")
app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec())