# stroke-risk-ml-classifier

A PyTorch model for determining stroke risk, based on established risk factors documented in leading medical textbooks, research papers, and guidelines from health organizations. The model achieves an impressive 99.96% accuracy on test data.

[Download the model here](stroke_risk_classifier_model.pth)

[Kaggle Dataset link](https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset/)

## Overview

This project uses PyTorch to build a neural network classifier for stroke risk prediction. The model analyzes several medical predictor variables to determine if a patient is at risk of a stroke.

## Dataset

The dataset uses 16 health indicators:

1. Chest Pain (binary)
2. Shortness of Breath (binary)
3. Irregular Heartbat (binary)
4. Fatigue & Weakness (binary)
5. Dizziness (binary)
6. Swelling (Edema) (binary)
7. Pain in Neck/Jaw/Shoulder/Back (binary)
8. Excessive Sweating	(binary)
9. Persistent Cough	(binary)
10. Nausea/Vomiting	(binary)
11. High Blood Pressure	(binary)
12. Chest Discomfort (Activity)	(binary)
13. Cold Hands/Feet (binary)	
14. Snoring/Sleep Apnea	(binary)
15. Anxiety/Feeling of Doom	(binary)
16. Age (integer)

To calculate:

1. Stroke Risk (high or low risk) (binary)

## Requirements

- Python 3.8+
- PyTorch
- pandas
- scikit-learn
- matplotlib

## Usage

1. Clone the repository
2. Create a virtual environment `py -m venv .venv` and activate it `.venv/Scripts/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the model: `python main.py`

## Model Architecture

- Input layer: 16 features
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 28 neurons with ReLU activation
- Output layer: 2 neurons (Binary classification)
- Optimization: Adam optimizer with learning rate 0.005
- Loss function: Cross Entropy Loss

## Results

The model is trained for 1000 epochs and the training progress can be visualized through a loss plot that is automatically generated and saved as 'loss_plot.png'.
The model had an accuracy of approximately 99.96%, though results may vary.

## License

[MIT License](LICENSE)
