from setuptools import setup, find_packages

setup(
    name="fraud_detection",
    version="0.1.0",
    packages=find_packages(),
    description="Fraud detection system for e-commerce and credit card transactions",
    author="10 Academy",
    author_email="info@10academy.org",
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "jupyter>=1.0.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.3.0",
        "shap>=0.45.0",
        "imbalanced-learn>=0.11.0",
    ],
    python_requires=">=3.8",
)