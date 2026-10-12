# Linear Regression Model

An interactive PySide6 desktop application for exploring multiple linear regression using local CSV datasets.

## What it does

1. Allows users to load a local CSV dataset.
2. Displays basic dataset information, including the number of rows and columns.
3. Lets users select one or more numeric input features and a target variable.
4. Calculates regression coefficients using NumPy's least-squares method.
5. Displays the regression equation, intercept, coefficients, and R² score.

## Technologies Researched

| Technology | Role | Why it fits the project |
|---|---|---|
| Python | Application logic | Handles data processing and regression calculations. |
| PySide6 | Desktop UI | Creates the interactive desktop application. |
| pandas | Dataset processing | Loads and organizes CSV data. |
| NumPy | Regression calculations | Performs numerical operations and least-squares calculations. |

## Run It

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install PySide6 pandas numpy
```

Launch the application:

```bash
python3 main.py
```

## Project Scope

This milestone focuses on a local multiple linear regression prototype. The application uses NumPy to calculate regression coefficients without relying on scikit-learn or SciPy.

The current implementation evaluates the model using the training data. Future improvements could include train/test splitting, predictions for new observations, and additional regression diagnostics.
