"""
Model Deployer

Packages a trained model into a deployable REST API service.
"""
import os
import pickle
from typing import Any, Dict


DEPLOY_TEMPLATE = '''
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{'status': 'ok'}})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {{}}
    features = data.get('features', [])
    try:
        X = pd.DataFrame(features)
        preds = model.predict(X)
        return jsonify({{'success': True, 'predictions': preds.tolist()}})
    except Exception as e:
        return jsonify({{'success': False, 'error': str(e)}}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
'''


def generate_deploy_package(model: Any, output_dir: str = 'deploy') -> Dict[str, str]:
    """Generate a deployable package with model + API + Dockerfile."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # API server
    api_path = os.path.join(output_dir, 'app.py')
    with open(api_path, 'w', encoding='utf-8') as f:
        f.write(DEPLOY_TEMPLATE.strip())
    
    # Requirements
    req_path = os.path.join(output_dir, 'requirements.txt')
    with open(req_path, 'w', encoding='utf-8') as f:
        f.write('flask>=2.0\nnumpy>=1.20\npandas>=1.3\nscikit-learn>=1.0\n')
    
    # Dockerfile
    docker_path = os.path.join(output_dir, 'Dockerfile')
    with open(docker_path, 'w', encoding='utf-8') as f:
        f.write('''FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
''')
    
    # README
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('''# Model Deployment

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

## Test prediction
```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [[1.0, 2.0, 3.0]]}'
```

## Docker
```bash
docker build -t model-api .
docker run -p 8000:8000 model-api
```
''')
    
    return {
        'model': model_path,
        'api': api_path,
        'requirements': req_path,
        'dockerfile': docker_path,
        'readme': readme_path,
    }
