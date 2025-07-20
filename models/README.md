# Models

This directory stores trained machine learning models and related artifacts.

## Model Naming Convention

Models should be named using the following convention:

```
{dataset_name}_{model_type}_{timestamp}.{extension}
```

For example:
- `creditcard_randomforest_20230615.pkl`
- `ecommerce_xgboost_20230616.joblib`

## Model Metadata

Each model should have an accompanying JSON file with the same name containing metadata:

```json
{
  "model_name": "creditcard_randomforest_20230615",
  "model_type": "RandomForestClassifier",
  "training_date": "2023-06-15",
  "features": ["feature1", "feature2", "..."]
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "metrics": {
    "accuracy": 0.99,
    "precision": 0.95,
    "recall": 0.92,
    "f1": 0.93,
    "auc": 0.98
  },
  "preprocessing": {
    "scaling": "StandardScaler",
    "encoding": "OneHotEncoder",
    "imputation": "SimpleImputer"
  }
}
```

## Model Versioning

When updating models, create a new file rather than overwriting existing ones. This ensures reproducibility and allows for model comparison.
