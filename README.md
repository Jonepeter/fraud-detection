# Fraud Detection Project

## Overview

This project aims to build robust machine learning models to detect fraudulent transactions in both e-commerce and bank credit card data. The solution leverages advanced data analysis, feature engineering, and model explainability techniques to improve fraud detection accuracy and support business decision-making.

## Business Need

Financial fraud is a critical challenge for e-commerce and banking platforms. Effective fraud detection reduces financial losses and builds trust with customers and partners. This project addresses:

- Class imbalance in fraud data
- Real-time and batch fraud detection
- Model interpretability for business trust

## Datasets

- **Fraud_Data.csv**: E-commerce transaction data
- **IpAddress_to_Country.csv**: Maps IP address ranges to countries
- **creditcard.csv**: Bank credit card transaction data

## Tools Used

This project leverages a variety of Python libraries and tools for data analysis, preprocessing, feature engineering, modeling, and evaluation:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms, preprocessing, and model evaluation
- **imbalanced-learn**: Handling class imbalance with techniques like SMOTE and undersampling
- **matplotlib** & **seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development and analysis
- **ipykernel**: Jupyter kernel for running Python code
- **Python 3.8+**: Core programming language

Additional tools and libraries may be used for model explainability (e.g., SHAP, LIME) and reporting as the project evolves.

## Key Features

- Data cleaning, preprocessing, and merging
- Feature engineering (time-based, frequency, geolocation)
- Handling class imbalance (SMOTE, undersampling, etc.)
- Model building: Logistic Regression, Random Forest/Gradient Boosting
- Model evaluation: AUC-PR, F1-Score, Confusion Matrix
- Model explainability: SHAP plots and interpretation

## Project Structure

```bash
  📁fraud-detection
   └── 📁data
         └── 📁processed
         └── 📁raw
            ├── creditcard.csv
            ├── Fraud_Data.csv
            ├── IpAddress_to_Country.csv
   └── 📁models
         ├── README.md
   └── 📁notebooks
         ├── 1_EDA.ipynb
         ├── 2_Preprocessing.ipynb
         ├── 3_Model_Building.ipynb
         ├── 4_Model_Explainability.ipynb
   └── 📁reports
         ├── README.md
   └── 📁src
         ├── download_data.py
         ├── eda.py
         ├── feature_engineering.py
         ├── model.py
         ├── preprocessing.py
   ├── .gitignore
   ├── README.md
   └── requirements.txt
```

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repo-url>
   cd fraud-detection
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add data files:**
   Place `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` in the `data/` folder.

## Usage

## Quick Start

```bash
# Install dependencies
make requirements

# Run complete pipeline (all tasks)
make run-all

# Run specific tasks
make run-task1  # Data Analysis and Preprocessing
make run-task2  # Model Building and Training
make run-task3  # Model Explainability

# Run tests
make test
```

## Complete Pipeline Usage

```python
from src.complete_pipeline import CompleteFraudPipeline

# Initialize pipeline
pipeline = CompleteFraudPipeline()

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    fraud_path='data/raw/Fraud_Data.csv',
    ip_path='data/raw/IpAddress_to_Country.csv',
    credit_path='data/raw/creditcard.csv'
)

# Access results
print(f"Best models: {pipeline.best_models}")
```

## Task-Specific Usage

```bash
# Run all tasks
python src/main.py --task all

# Run individual tasks
python src/main.py --task 1  # Data preprocessing
python src/main.py --task 2  # Model training
python src/main.py --task 3  # Model explainability
```

## Pipeline Features

### Task 1: Data Analysis and Preprocessing
- Handle missing values and duplicates
- Convert IP addresses to integer format
- Merge datasets for geolocation analysis
- Create time-based features (hour_of_day, day_of_week, time_since_signup)
- Transaction frequency and velocity features
- Handle class imbalance with SMOTE
- Normalization and scaling
- Categorical encoding

### Task 2: Model Building and Training
- Train-test split with stratification
- Multiple models: Logistic Regression, Random Forest, XGBoost
- Evaluation with imbalanced data metrics (AUC-PR, F1-Score)
- Best model selection based on AUC-PR

### Task 3: Model Explainability
- SHAP analysis for best-performing models
- Global and local feature importance
- Summary plots and feature rankings

## Contribution Guidelines

1. Fork the repository and create a new branch for your feature or bugfix.
2. Follow PEP8 and write clear, modular code with docstrings.
3. Add tests or notebook examples for new features.
4. Submit a pull request with a clear description of your changes.

## License

This project is for educational purposes. For commercial use, please contact the project owner.

## Acknowledgements

- 10 Academy
- Adey Innovations Inc.
- Open-source contributors and dataset providers