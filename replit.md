# World of Warships Offset X Predictor

## Overview
This is a machine learning-powered web application that predicts horizontal offset (offset_x) for aiming in World of Warships. The application uses a Random Forest Regression model trained on real game data to help players aim more consistently with dynamic crosshairs like the **Nomogram Classic Top Web** by stiv32.

## Project Status
**Status**: ✅ Fully functional and ready to use  
**Last Updated**: November 27, 2025

## Architecture

### Tech Stack
- **Backend**: FastAPI (Python 3.11)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **ML Model**: Random Forest Regressor (scikit-learn)
- **Deployment**: Replit Autoscale

### Project Structure
```
.
├── main.py                 # FastAPI application (backend + API)
├── static/                 # Frontend web interface
│   ├── index.html         # Main web page
│   ├── style.css          # Styling
│   └── script.js          # Client-side logic
├── offset_predictor.pkl   # Trained ML model
├── train_model.py         # Model training script
├── data_tembak.csv        # Training dataset
├── requirements.txt       # Python dependencies
├── GUI.py                 # Desktop GUI (Tkinter - for local use only)
└── API/                   # Legacy API folder
```

## Features

### Web Interface
- **Modern UI**: Responsive web interface with gradient backgrounds and smooth animations
- **Real-time Predictions**: Submit shell travel time, distance, and angle to get instant offset predictions
- **User-friendly**: Clear input validation and error handling
- **Mobile-friendly**: Responsive design that works on all devices

### API
- **POST /api/predict**: Predict offset_x from input parameters
- **GET /api/health**: Health check endpoint
- **GET /**: Serves the web interface

### ML Model
- **Algorithm**: Random Forest Regression
- **Features**: 9 engineered features including trigonometric transformations
- **Performance**: MAE ~0.915, RMSE ~1.167
- **Input Variables**:
  - Shell travel time (seconds)
  - Distance to target (km)
  - Angle between aim and enemy movement (degrees)

## How to Use

### Web Interface
1. Open the application in your browser
2. Enter the **shell travel time** from your binocular view
3. Input the **distance** to the enemy ship
4. Enter the **angle** between your aim and the enemy's movement direction
5. Click "Calculate Offset" to get your prediction
6. Use the predicted offset value with your Nomogram Classic Top Web crosshair

### API Usage
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "shell_travel_time": 8.65,
    "distance": 13.55,
    "angle": 74
  }'
```

Response:
```json
{
  "offset_x": 5,
  "offset_x_raw": 5.3571407897657926
}
```

## Development

### Running Locally
The FastAPI server runs automatically on port 5000:
```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

### Retraining the Model
If you add more training data to `data_tembak.csv`, retrain the model:
```bash
python train_model.py
```

This will generate a new `offset_predictor.pkl` file. Restart the server to use the updated model.

## Dataset Format
The training data (`data_tembak.csv`) contains:
- `shell_travel_time`: Time for shell to reach target
- `distance`: Distance to enemy ship (km)
- `angle`: Angle between aim and enemy movement (degrees)
- `enemy_max_speed`: Maximum speed of enemy ship (knots)
- `enemy_actual_speed`: Calculated actual speed (optional)
- `offset_x`: Horizontal offset in binocular crosshair (target variable)

## Technical Details

### Feature Engineering
The model uses these engineered features:
1. `shell_travel_time` - Direct input
2. `distance` - Direct input
3. `angle` - Direct input (degrees)
4. `angle_rad` - Angle converted to radians
5. `sin_a` - Sine of angle
6. `cos_a` - Cosine of angle
7. `dist_time` - Distance × shell travel time
8. `time_sin` - Shell travel time × sin(angle)
9. `time_cos` - Shell travel time × cos(angle)

### Model Parameters
- **Best hyperparameters**: max_depth=5, min_samples_leaf=5, n_estimators=100
- **Cross-validation**: 3-fold CV
- **Random state**: 42 (for reproducibility)

## Deployment
This application is configured for Replit Autoscale deployment:
- Automatically scales based on traffic
- Runs on-demand (no constant server costs when idle)
- Production-ready FastAPI server with uvicorn

## Important Notes
- This model provides predictions based on training data, not full physics simulation
- Results may vary based on the quality and quantity of training data
- Best used as an aim assist tool, not a replacement for player skill
- The desktop GUI (GUI.py) requires a local Python environment with tkinter

## Credits
- Model training approach based on the project README
- Designed for use with **Nomogram Classic Top Web** crosshair by stiv32
- Created for World of Warships players
