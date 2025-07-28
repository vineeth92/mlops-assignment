import joblib
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

# Load the trained scikit-learn model
model = joblib.load("models/sklearn_model.joblib")
weights = model.coef_
bias = model.intercept_

# Save unquantized parameters
unquant_params = {"weights": weights, "bias": bias}
joblib.dump(unquant_params, "models/unquant_params.joblib")

# Manual quantization
w_min, w_max = weights.min(), weights.max()
scale = 255 / (w_max - w_min)
quantized_weights = np.round((weights - w_min) * scale).astype(np.uint8)
quantized_bias = np.round((bias - w_min) * scale).astype(np.uint8)

quant_params = {
    "weights": quantized_weights,
    "bias": quantized_bias,
    "scale": scale,
    "min": w_min
}
joblib.dump(quant_params, "models/quant_params.joblib")

# Dequantize
dequantized_weights = (quantized_weights.astype(np.float32) / scale) + w_min
dequantized_bias = (quantized_bias.astype(np.float32) / scale) + w_min

# Load dequantized values into PyTorch model
input_size = dequantized_weights.shape[0]
model_torch = torch.nn.Linear(input_size, 1)
model_torch.weight.data = torch.tensor([dequantized_weights], dtype=torch.float32)
model_torch.bias.data = torch.tensor([dequantized_bias], dtype=torch.float32)

# Run inference
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
predicted = model_torch(X_test_tensor).detach().numpy()

# Compare R² scores
original_preds = model.predict(X_test)
print("Original R² Score:", r2_score(y_test, original_preds))
print("Quantized R² Score:", r2_score(y_test, predicted))

