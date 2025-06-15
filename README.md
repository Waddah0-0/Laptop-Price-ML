# Laptop Price Prediction Model

A machine learning model that predicts laptop prices based on various specifications using XGBoost algorithm. This project demonstrates different approaches to data preprocessing, feature engineering, and model training.

## Features
- Comprehensive data preprocessing pipeline
- Multiple approaches for handling missing values
- Different encoding methods for categorical variables
- XGBoost model implementation
- Model evaluation metrics and visualizations
- Model persistence and loading functionality

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- joblib

## Installation
1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure
```
laptop-price-prediction/
├── saved_models/           # Saved model and preprocessing objects
├── data/                   # Dataset directory
├── model.py               # Main model implementation
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Implementation Details

### Data Preprocessing

#### 1. Handling Missing Values
Three approaches are implemented:
- **Drop rows with missing values**:
  ```python
  x_train.dropna(inplace=True)
  x_test.dropna(inplace=True)
  ```
- **Fill with mean/median/mode**:
  ```python
  imputer = SimpleImputer(strategy="mean")
  x_train[num_col] = imputer.fit_transform(x_train[num_col])
  ```
- **Fill and mark missing values**:
  - Fill missing values and add a new column indicating if the value was imputed
  - Useful for tracking which values were originally missing

#### 2. Categorical Variable Encoding
Multiple approaches are available:
- **Drop categorical columns**:
  ```python
  reduced_x_train = x_train.select_dtypes(exclude=["object"])
  ```
- **One-Hot Encoding**:
  ```python
  encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
  encoded_x_train_cols = pd.DataFrame(encoder.fit_transform(x_train[low_cardinality_col]))
  ```
- **Label Encoding** (commented in code)
  - Alternative approach for ordinal categorical variables

### Model Training

#### XGBoost Implementation
```python
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
model.fit(encoded_x_train, y_train)
```

### Model Evaluation
Multiple metrics are used:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared Score
- Mean Percentage Error

### Visualization
- Scatter plot of actual vs predicted prices
- Error distribution histogram
- Correlation analysis

## Usage

### 1. Train and Save the Model
```python
python model.py
```

### 2. Make Predictions on New Data
```python
from model import load_and_predict
import pandas as pd

# Load new data
new_data = pd.read_csv('path_to_new_data.csv')

# Make predictions
predictions = load_and_predict(new_data)
print('Predicted prices:', predictions)
```

## Model Performance
The model is evaluated using multiple metrics to ensure comprehensive performance assessment:
- Mean Absolute Error (MAE): Average absolute difference between predicted and actual prices
- Root Mean Squared Error (RMSE): Square root of the average squared differences
- R-squared Score: Proportion of variance in prices explained by the model
- Mean Percentage Error: Average percentage difference between predicted and actual prices

## Data Preprocessing Details
- **Missing Values**: Handled using mean imputation by default
- **Categorical Variables**: Encoded using OneHotEncoder
- **Feature Selection**: Includes both numeric and categorical features
- **Data Transformation**: Preserves original data structure while enabling model compatibility

## Model Details
- **Algorithm**: XGBoost
- **Features**: Various laptop specifications
- **Target**: Price
- **Evaluation**: Multiple metrics for comprehensive assessment
- **Persistence**: Model and preprocessing objects are saved for future use

## Alternative Approaches
The code includes commented sections showing alternative approaches for:
1. Missing value handling
2. Categorical variable encoding
3. Feature selection
4. Model evaluation

## Author
Waddah

## Acknowledgments
- Dataset source
- Any other acknowledgments 