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

- **Exploratory Data Analysis (EDA):**
  - Run EDA scripts in `src/eda.py` or use Jupyter notebooks in `notebooks/`.
  - Example:
  
    ```bash
    python src/eda.py
    ```

- **Model Training & Evaluation:**
  - Scripts for feature engineering, model training, and evaluation will be in `src/`.
  - Notebooks provide step-by-step analysis and results.

## Key Features

- Data cleaning, preprocessing, and merging
- Feature engineering (time-based, frequency, geolocation)
- Handling class imbalance (SMOTE, undersampling, etc.)
- Model building: Logistic Regression, Random Forest/Gradient Boosting
- Model evaluation: AUC-PR, F1-Score, Confusion Matrix
- Model explainability: SHAP plots and interpretation

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