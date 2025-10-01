# Face Analysis API

## Overview
 comprehensive RESTful API for Alzheimer's Disease (AD) assessment combining facial asymmetry analysis and cognitive evaluation using computer vision and machine learning techniques.

## Features
- 🔍 **Automated face detection** using MediaPipe FaceMesh (468 landmarks)
- 📊 **Dual ML predictions** powered by XGBoost models:
  - **6QDS Cognitive Assessment** based on questionnaire responses
  - **Facial Asymmetry Classification** from facial landmark analysis
- 🖼️ **Visual analysis** with marked facial landmarks and symmetry lines
- 📦 **Multi-format support** for compressed image archives
- 📝 **Integrated questionnaire processing** for comprehensive assessment

## Input Requirements

### 1. Compressed Image Archive
Upload a compressed archive containing facial photographs:
- **Supported formats**: `.zip`, `.7z`, `.rar`
- **File size limit**: 50MB
- **Image formats**: JPG, JPEG, PNG, BMP, TIFF
- **Recommended**: 5-20 front-facing photos for optimal accuracy

### 2. Questionnaire Data
Provide demographic and assessment questionnaire responses:
- **age**: Age in years
- **gender**: Gender (0: Female, 1: Male)
- **education_years**: Years of education
- **q1-q10**: Questionnaire responses (10 questions)

## Output
Returns a JSON response with comprehensive analysis results:

```json
{
  "success": true,
  "error": null,
  "q6ds_classification_result": 0.75,
  "asymmetry_classification_result": 0.85,
  "marked_figure": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

### Response Fields
| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Analysis completion status |
| `error` | string/null | Error message if analysis failed |
| `q6ds_classification_result` | float/null | 6QDS cognitive assessment prediction (0.0-1.0) |
| `asymmetry_classification_result` | float/null | Facial asymmetry ML prediction (0.0-1.0) |
| `marked_figure` | string/null | Base64-encoded image with facial landmarks |


## API Usage Examples

### Using curl
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@face_photos.zip" \
     -F "age=70" \
     -F "gender=1" \
     -F "education_years=12" \
     -F "q1=1" \
     -F "q2=0" \
     -F "q3=1" \
     -F "q4=1" \
     -F "q5=0" \
     -F "q6=1" \
     -F "q7=0" \
     -F "q8=1" \
     -F "q9=0" \
     -F "q10=1"
```

### Using Python requests
```python
import requests

# Prepare questionnaire data
questionnaire_data = {
    "age": 70,
    "gender": 1,  # Male
    "education_years": 12,
    "q1": 1, "q2": 0, "q3": 1, "q4": 1, "q5": 0,
    "q6": 1, "q7": 0, "q8": 1, "q9": 0, "q10": 1
}

# Upload file with questionnaire data
with open("face_photos.zip", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
        data=questionnaire_data
    )
    result = response.json()
    print(result)
```


## Project Structure
```
api/
├── data/
│   ├── .gitignore
│   ├── haarcascade_frontalface_default.xml
│   ├── symmetry_all_pairs.csv  # Facial symmetry mapping
│   ├── xgb_face_asym_model.csv  # Facial symmetry mapping
│   └── xgb_6qds_model.json            # Pre-trained ML model
├── .gitignore                  
├── main.py                     # Application entry point
├── poetry.lock                 # Locked dependency versions
├── pyproject.toml              # Poetry dependencies
└── README.md
```

## API Endpoints
- `POST /analyze` - Upload photos and questionnaire data for comprehensive analysis
- `GET /health` - Service health check with supported formats
- `GET /docs` - Interactive Swagger API documentation
- `GET /redoc` - Alternative ReDoc API documentation
- `GET /` - API information and configuration details

## Docker Quick Start

### change path
```bash
cd path/to/AD-Sensor-Project
```

### build image and named face-analysis-api
```bash
docker build -t face-analysis-api .
```

### start api
```bash
docker-compose up -d
```

## Development

### Prerequisites
- Python 3.11.x
- Intel RealSense SDK
- Arduino IDE (for LED functionality)
- Git

### Setup Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/a7266165/AD-Sensor-Project.git
   cd AD-Sensor-Project
   ```

2. **Install Poetry for dependency management** (Skip if already installed)
   ```bash
   # Create poetry environment
   conda create -n poetry python=3.11
   
   # Activate the poetry environment
   conda activate poetry
   
   # Install poetry in this environment
   pip install poetry
   
   # Add poetry to PATH (optional but recommended)
   # For Windows:
   # Add the poetry installation path to your system PATH environment variable
   # Typical path: C:\Users\[YourUsername]\anaconda3\envs\poetry\Scripts
   
   # For macOS/Linux:
   # Add to your ~/.bashrc or ~/.zshrc:
   # export PATH="$HOME/anaconda3/envs/poetry/bin:$PATH"
   ```

3. **Create work environment and install dependencies**
   ```bash
   # Create project environment
   conda create -n env_AD_api python=3.11

   # Activate project environment
   conda activate env_AD_api

   # Using Poetry to install project dependencies 
   poetry install --no-root

### Running in Development Mode
```bash
# Start with auto-reload
python main.py

# Access interactive documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### API Testing
Visit `http://localhost:8000/docs` for interactive API testing with Swagger UI, where you can:
- Upload test image archives
- Input questionnaire responses
- View real-time analysis results
- Download marked facial images


## Hardware Requirements

- **Camera**: Intel RealSense D435/D415 or compatible
- **Arduino**: Arduino Uno/Mega with LED circuit (optional)
- **Computer**: not decide