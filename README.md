# Fraud Detection Project

A comprehensive machine learning pipeline for detecting fraudulent transactions using e-commerce and credit card transaction datasets.

## Features

### Data Preprocessing
- **Missing Value Handling**: Imputation for numerical and categorical features
- **Data Cleaning**: Duplicate removal, data type corrections
- **Feature Engineering**: 
  - Time-based features (hour_of_day, day_of_week, time_since_signup)
  - Transaction frequency and velocity
  - IP geolocation mapping
- **Class Imbalance Handling**: SMOTE oversampling, random undersampling
- **Feature Scaling**: StandardScaler normalization
- **Categorical Encoding**: Label encoding

### Model Training
- **Logistic Regression**: Interpretable baseline
- **Random Forest**: Ensemble with feature importance
- **XGBoost**: High-performance gradient boosting
- **LightGBM**: Fast gradient boosting

### Evaluation Metrics
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve (for imbalanced data)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed results
- **Cross-Validation**: Stratified k-fold

### Model Explainability
- **SHAP Values**: Shapley Additive exPlanations
- **Summary, Force, and Waterfall Plots**: Global and local explanations
- **Feature Importance Rankings**

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd fraud-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your data files to the `data/` directory:
   - `Fraud_Data.csv`
   - `IpAddress_to_Country.csv`
   - `creditcard.csv`

## Usage

### Run Complete Pipeline
```bash
python src/pipeline.py
```

### Run Exploratory Data Analysis
```bash
python notebooks/exploratory_analysis.py
```

### Use Individual Components
```python
from src.data_preprocessing import DataPreprocessor
from src.models import FraudModels
from src.explainability import ModelExplainer

# Initialize
preprocessor = DataPreprocessor()
model_trainer = FraudModels()

# Load and preprocess data
fraud_data, ip_data, credit_data = preprocessor.load_data(...)
X, y = preprocessor.preprocess_fraud_data(...)

# Train models
results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)

# Explain best model
explainer = ModelExplainer(best_model, X_train, X_test, feature_names)
explainer.generate_explanation_report('results/')
```

## Key Challenges Addressed

### Class Imbalance
- **Problem**: Fraudulent transactions are rare (<1% of data)
- **Solution**: SMOTE oversampling, AUC-PR prioritization

### Feature Engineering
- **Time Features**: Extract temporal patterns
- **Behavioral Features**: User/device transaction patterns
- **Geolocation**: Map IPs to countries

### Model Selection
- **Baseline**: Logistic Regression
- **Advanced**: Random Forest, XGBoost, LightGBM
- **Evaluation**: Stratified cross-validation

## Results

The pipeline generates:
- **Model Performance**: Comparison of all models
- **Feature Importance**: SHAP-based rankings
- **Visualizations**: Summary, force, and waterfall plots
- **Final Report**: Comprehensive analysis

## Model Interpretability

### SHAP Explanations
- **Global Importance**: Key features overall
- **Local Explanations**: Individual predictions
- **Feature Interactions**: How features work together

### Key Insights
- Most important fraud indicators
- Temporal and geographic risk patterns
- User behavior anomalies

## Best Practices Implemented

1. **Pipeline Architecture**: Modular, reusable
2. **Class Imbalance**: Proper sampling and metrics
3. **Feature Engineering**: Domain-specific
4. **Model Validation**: Stratified cross-validation
5. **Interpretability**: SHAP explanations
6. **Code Quality**: Clean, documented

## Extending the Project

### Add New Models
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

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- lightgbm
- shap
- matplotlib, seaborn

## Project Structure

```
└── 📁fraud-detection
    └── 📁.github
        └── 📁workflows
            ├── python-app.yml                 # CI/CD pipeline configuration
    └── 📁data
        └── 📁processed                        # Cleaned and processed datasets
        └── 📁raw                             # Original datasets (Fraud_Data.csv, creditcard.csv, IpAddress_to_Country.csv)
    └── 📁models
        ├── README.md                         # Model storage guidelines
    └── 📁notebooks
        ├── complete_pipeline_demo.ipynb      # Interactive pipeline demonstration
    └── 📁reports
        ├── final_report.md                   # Final analysis report
        ├── README.md                         # Report documentation
    └── 📁src
        ├── __init__.py                       # Package initialization
        ├── complete_pipeline.py              # Main pipeline with all 3 tasks
        ├── load_data.py                      # Data loading utilities
        ├── main.py                           # CLI entry point
        ├── utils.py                          # Helper functions
    └── 📁tests
        ├── __init__.py                       # Test package initialization
        ├── test_complete_pipeline.py         # Pipeline tests
        ├── test_load_data.py                 # Data loading tests
        ├── test_utils.py                     # Utility function tests
        ├── README.md                         # Testing documentation
    ├── .gitignore                            # Git ignore patterns
    ├── config.yaml                           # Pipeline configuration
    ├── Makefile                              # Build automation
    ├── README.md                             # Project documentation
    ├── requirements.txt                      # Python dependencies
    ├── run_pipeline.py                       # Quick pipeline runner
    └── setup.py                              # Package installation
```

### Key Components:
- **complete_pipeline.py**: Contains all three tasks (data preprocessing, model training, explainability)
- **main.py**: Command-line interface for running specific tasks or complete pipeline
- **run_pipeline.py**: Simple script to execute the entire pipeline
- **config.yaml**: Configuration file for model parameters and data paths
- **tests/**: Comprehensive test suite for all components

## Setup Notes

- Python 3.7+ recommended
- Use a virtual environment (e.g., `venv` or `conda`)

## License

This project is for educational purposes. Ensure you have proper permissions for any datasets used.