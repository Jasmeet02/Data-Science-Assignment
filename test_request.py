import requests, json

with open("sample_request.json") as f:
    payload = json.load(f)

response = requests.post("http://127.0.0.1:5000/predict", json=payload)
print(response.json())
