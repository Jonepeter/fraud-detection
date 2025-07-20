# Tests

This directory contains unit tests for the fraud detection project.

## Running Tests

To run all tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=src tests/
```

To generate a coverage report:

```bash
pytest --cov=src --cov-report=html tests/
```

## Test Structure

- `test_load_data.py`: Tests for data loading functionality
- `test_preprocessing.py`: Tests for data preprocessing functions
- `test_feature_engineering.py`: Tests for feature engineering functions
- `test_model.py`: Tests for model training and evaluation
- `test_utils.py`: Tests for utility functions

## Adding New Tests

When adding new functionality to the project, please add corresponding tests. Follow these guidelines:

1. Create a new test file if testing a new module
2. Use descriptive test names that explain what is being tested
3. Use fixtures for common test setup
4. Mock external dependencies when appropriate
5. Test both normal and edge cases

## Test Data

Test data files should be placed in the `tests/test_data` directory. These files should be small and focused on testing specific functionality.