import requests
import json

# API endpoint
base_url = "http://localhost:8080"

# Test health check
print("Testing health check...")
response = requests.get(f"{base_url}/health")
print(f"Health check: {response.json()}")

# Test single prediction
print("\nTesting single prediction...")
test_data = {
    "features": [1.2, -0.5, 0.8, 2.1]
}

response = requests.post(f"{base_url}/predict", json=test_data)
print(f"Single prediction: {response.json()}")

# Test batch prediction
print("\nTesting batch prediction...")
batch_data = {
    "features_list": [
        [1.2, -0.5, 0.8, 2.1],
        [0.3, 1.1, -0.2, 1.5],
        [-0.8, 0.9, 1.3, -0.7]
    ]
}

response = requests.post(f"{base_url}/batch_predict", json=batch_data)
print(f"Batch prediction: {json.dumps(response.json(), indent=2)}")
