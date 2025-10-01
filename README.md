# Intruder-Detection

This project is a real-time surveillance system that uses computer vision to automatically detect human intruders and generate visual alerts.

## Features
- **Two-Stage Detection:** A lightweight motion detector pre-screens for movement, and a powerful CNN verifies human presence for optimal performance.
- **Deep Learning Model:** Utilizes a ResNet-18 model fine-tuned on the INRIA Person Dataset for high accuracy.
- **Real-Time Alerts:** Provides immediate on-screen visual feedback with bounding boxes and an alert banner.
- **Modular Code:** The project is cleanly separated into `train.py` for model training and `surveillance.py` for live deployment.

## Technology Stack
- Python 3
- PyTorch
- OpenCV
- NumPy

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jeevada/Intruder-Detection.git
    cd Intruder-Detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Download the Dataset
Before training, you must download the INRIA Person Dataset. Place the extracted `INRIAPerson` folder in the root of the project directory.

### 2. Train the Model
Run the training script. This will train the model on the dataset and save the weights as `person_detector_model.pth`.
```bash
python train.py
```

### 3. Run the Surveillance System
Once the model is trained, run the live surveillance application.
```bash
python surveillance.py
```

Press 'q' to exit the application.
