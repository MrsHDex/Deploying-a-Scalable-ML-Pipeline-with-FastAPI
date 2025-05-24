import pytest
# TODO: add necessary import
import numpy as np 
from ml.model import train_model, compute_model_metrics, inference

# TODO: implement the first test. Change the function name and input as needed
def test_model_predict_output_shape():
    """
    Test that the trained model produces predictions of the expected shape.
    """
    X = np.random.rand(20, 4)
    y = np.random.randint(0, 2, size=20)
    model = train_model(X, y)

    preds = model.predict(X)
    assert preds.shape == (20,), "Prediction output shape should match number of samples."
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions should be binary (0 or 1)."


# TODO: implement the second test. Change the function name and input as needed
def test_inference_function():
    """
    Test the inference function returns correct shape and valid binary predictions.
    """
    X = np.random.rand(10, 4)
    y = np.random.randint(0, 2, size=10)
    model = train_model(X, y)

    preds = inference(model, X)
    assert preds.shape == (10,), "Inference output shape should match input."
    assert set(np.unique(preds)).issubset({0, 1}), "Inference results should be binary."


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test compute_model_metrics returns valid precision, recall, and f1 score.
    """
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    for metric in (precision, recall, f1):
        assert isinstance(metric, float), "Metrics should be floats."
        assert 0.0 <= metric <= 1.0, "Metric values should be between 0 and 1."
