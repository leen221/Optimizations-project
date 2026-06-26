# IRIS Neural Network Classifier
## How to Run

### Requirements
- Python 3.8+
- numpy
- pandas
- matplotlib

### Install Requirements
```
pip install numpy pandas matplotlib
```

### Files Needed
```
iris_classifier.py    ← main code
iris_training.csv     ← training data
iris_test.csv         ← test data
```

### Run
```
python iris_classifier.py
```

### Output
- Printed table with accuracy and MSE for all 3 models
- Confusion matrices in console
- `iris_results.png` — visual dashboard with plots

### Models Compared
1. No Regularization  → baseline
2. LASSO (L1)         → λ = 0.001
3. Ridge (L2)         → λ = 0.001
