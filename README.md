# Fraud Detection Project

A comprehensive machine learning pipeline for detecting fraudulent transactions using two datasets: e-commerce fraud data and credit card transaction data.


## Features

### Data Preprocessing
- **Missing Value Handling**: Imputation strategies for numerical and categorical features
- **Data Cleaning**: Duplicate removal and data type corrections
- **Feature Engineering**: 
  - Time-based features (hour_of_day, day_of_week, time_since_signup)
  - Transaction frequency and velocity features
  - IP geolocation mapping
- **Class Imbalance Handling**: SMOTE oversampling and random undersampling
- **Feature Scaling**: StandardScaler normalization
- **Categorical Encoding**: Label encoding for categorical variables

### Model Training
- **Logistic Regression**: Interpretable baseline model
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting for high performance
- **LightGBM**: Fast gradient boosting alternative

### Evaluation Metrics
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve (critical for imbalanced data)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results
- **Cross-Validation**: Stratified k-fold validation

### Model Explainability
- **SHAP Values**: Shapley Additive exPlanations
- **Summary Plots**: Global feature importance visualization
- **Force Plots**: Individual prediction explanations
- **Waterfall Plots**: Step-by-step prediction breakdown
- **Feature Importance Rankings**: Quantified feature contributions

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

1. Add your data files to the `data/` directory:
   - `Fraud_Data.csv`
   - `IpAddress_to_Country.csv`
   - `creditcard.csv`

## Usage

### Run Complete Pipeline
```bash
cd src
python pipeline.py
```

### Run Exploratory Data Analysis
```bash
cd notebooks
python exploratory_analysis.py
```

### Individual Components
```python
from src.data_preprocessing import DataPreprocessor
from src.models import FraudModels
from src.explainability import ModelExplainer

# Initialize components
preprocessor = DataPreprocessor()
model_trainer = FraudModels()

# Load and preprocess data
fraud_data, ip_data, credit_data = preprocessor.load_data(...)
X, y = preprocessor.preprocess_fraud_data()

# Train models
results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)

# Explain best model
explainer = ModelExplainer(best_model, X_train, X_test, feature_names)
explainer.generate_explanation_report('results/')
```

## Key Challenges Addressed

### Class Imbalance
- **Problem**: Fraudulent transactions are rare (typically <1% of data)
- **Solution**: SMOTE oversampling with careful evaluation using AUC-PR
- **Metrics**: Prioritize Precision-Recall AUC over ROC-AUC

### Feature Engineering
- **Time Features**: Extract temporal patterns from timestamps
- **Behavioral Features**: User and device transaction patterns
- **Geolocation**: Map IP addresses to countries for geographic analysis

### Model Selection
- **Baseline**: Logistic Regression for interpretability
- **Advanced**: Ensemble methods (Random Forest, XGBoost, LightGBM)
- **Evaluation**: Cross-validation with stratified sampling

## Results

The pipeline generates:
- **Model Performance**: Comparison of all models with key metrics
- **Feature Importance**: SHAP-based feature rankings
- **Visualizations**: Summary plots, force plots, and waterfall charts
- **Final Report**: Comprehensive analysis summary

## Model Interpretability

### SHAP Explanations
- **Global Importance**: Which features matter most overall
- **Local Explanations**: Why specific predictions were made
- **Feature Interactions**: How features work together

### Key Insights
The model explanations reveal:
- Most important fraud indicators
- Temporal patterns in fraudulent behavior
- Geographic risk factors
- User behavior anomalies

## Best Practices Implemented

1. **Pipeline Architecture**: Modular, reusable components
2. **Class Imbalance**: Appropriate sampling and metrics
3. **Feature Engineering**: Domain-specific transformations
4. **Model Validation**: Stratified cross-validation
5. **Interpretability**: SHAP-based explanations
6. **Code Quality**: Clean, documented, maintainable code

## Extending the Project

### Adding New Models
```python
# In models.py, add to initialize_models():
self.models['new_model'] = YourModel(parameters...)
```

### Custom Features
```python
# In data_preprocessing.py, extend feature_engineering_fraud():
df['new_feature'] = your_transformation(df)
```

### Additional Explanations
```python
# In explainability.py, add new visualization methods:
def plot_custom_explanation(self):
    # Your custom SHAP visualization
```

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning algorithms
- imbalanced-learn: Handling class imbalance
- xgboost: Gradient boosting
- lightgbm: Fast gradient boosting
- shap: Model explainability
- matplotlib/seaborn: Visualization

## Project Structure

```
fraud/
├── data/                          # Data directory (add your CSV files here)
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── creditcard.csv
├── src/                           # Source code
│   ├── config.py                  # Configuration settings
│   ├── data_preprocessing.py      # Data preprocessing pipeline
│   ├── models.py                  # Model training and evaluation
│   ├── explainability.py          # SHAP-based model explainability
│   └── pipeline.py                # Main pipeline orchestrator
├── notebooks/                     # Analysis notebooks
│   └── exploratory_analysis.py    # EDA script
├── models/                        # Saved models
├── results/                       # Results and explanations
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Setup Instructions

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd fraud
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add data files**:  
   Place your datasets (e.g., `Fraud_Data.csv`, `IpAddress_to_Country.csv`, `creditcard.csv`) in the `data/` directory.

4. **Run the pipeline**:
   ```bash
   python src/pipeline.py
   ```

5. **(Optional) Explore notebooks**:  
   Open and run the analysis notebook in the `notebooks/` directory for exploratory data analysis.

**Note:**  
- Python 3.7+ is recommended.
- For best results, use a virtual environment (e.g., `venv` or `conda`).

## License

This project is for educational purposes. Please ensure you have proper permissions for any datasets used.