# README

## Project Title: Chronic Kidney Disease Prediction with Neural Networks

### Overview
This project implements a binary classification system to predict chronic kidney disease (CKD) using a neural network model. The dataset is preprocessed with techniques for handling missing values, outlier management, and feature scaling. The model evaluates the effects of hidden layer neuron configurations and data standardization on classification accuracy.

### Dataset
The Chronic Kidney Disease dataset from the UCI Machine Learning Repository is used. It contains 400 samples with 24 features and one target variable.

#### Key Features:
- **Numerical Features:** Age, Blood Pressure, Blood Glucose, Hemoglobin, etc.
- **Categorical Features:** Red Blood Cells, Pus Cells, Appetite, etc.
- **Target Class:** Binary (CKD, Not CKD).

#### Data Preprocessing Steps:
1. **Handling Missing Values:**
   - Numerical columns filled with mean values.
   - Categorical columns filled with mode values.
2. **Outlier Management:**
   - Identified using the Interquartile Range (IQR).
   - Capped to within calculated bounds.
3. **Encoding Categorical Features:**
   - Ordinal Encoding for ordered categories.
   - One-Hot Encoding for nominal categories.
4. **Feature Scaling:**
   - Applied Yeo-Johnson transformation.
   - Standardized to zero mean and unit variance.

### Neural Network Architecture
- **Input Layer:** 24 neurons (equal to the number of features).
- **Hidden Layer:** 10 neurons.
- **Output Layer:** 1 neuron for binary classification.

#### Activation Functions:
- Sigmoid activation for both hidden and output layers.

### Implementation
1. **Forward Propagation:** Computes hidden and output layer activations.
2. **Backpropagation:** Updates weights and biases using the sigmoid derivative.
3. **Training:**
   - Runs for 1000 epochs.
   - Learning rate: 0.01.
4. **Prediction:** Classifies test data based on a threshold of 0.5.

### Model Evaluation
- **Metrics Used:** Accuracy and Confusion Matrix.
- **Results:**
  - Accuracy: 96.25%.
  - Confusion Matrix:
    ```
    [[50, 2],
     [ 1, 27]]
    ```

### Experiments and Observations
#### Number of Hidden Layer Neurons:
- Optimal size: 10-15 neurons.
- Larger sizes resulted in overfitting.

#### Data Standardization:
- Standardization improved accuracy and stability.
- Ensures faster convergence during training.

### Visualizations
- Boxplots and histograms for feature distribution.
- Confusion matrix heatmap for performance evaluation.
- Accuracy plots for neuron configurations with standardized vs. non-standardized data.

### Technologies Used
- **Programming Language:** Python.
- **Libraries:**
  - Pandas, NumPy, Matplotlib, Seaborn.
  - Scikit-learn (for preprocessing and metrics).

### How to Run
1. Clone the repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the main Python script:
   ```bash
   python main.py
   ```

### Future Work
- Experiment with additional neural network architectures.
- Explore other preprocessing techniques.
- Address class imbalance using SMOTE or similar techniques.

### Contributors
- Venushki Thilakawardana

### Acknowledgements
- Dataset provided by UCI Machine Learning Repository.
- Guidance from the Artificial Neural Networks & Evolutionary Computing course (CM4310).

