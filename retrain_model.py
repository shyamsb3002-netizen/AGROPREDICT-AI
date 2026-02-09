"""
Script to retrain the crop recommendation model with expanded crop dataset.
Includes 40+ crops with realistic parameter ranges based on Indian agriculture.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Define crop parameters (typical ranges for each crop)
# Format: (N_min, N_max, P_min, P_max, K_min, K_max, temp_min, temp_max, humidity_min, humidity_max, ph_min, ph_max, rainfall_min, rainfall_max)
crop_params = {
    # Cereals
    'rice': (60, 100, 40, 70, 35, 55, 20, 30, 80, 95, 5.5, 7.5, 150, 300),
    'wheat': (80, 120, 40, 65, 25, 45, 12, 22, 50, 70, 6.0, 7.5, 50, 100),
    'maize': (60, 100, 35, 60, 25, 45, 18, 28, 55, 75, 5.5, 7.5, 60, 120),
    'barley': (60, 90, 35, 55, 20, 40, 10, 20, 40, 60, 6.0, 8.0, 40, 80),
    'sorghum': (50, 80, 30, 50, 20, 40, 25, 35, 40, 60, 5.5, 8.0, 40, 100),
    'pearl_millet': (40, 70, 20, 40, 15, 35, 25, 35, 40, 60, 6.0, 8.0, 25, 60),
    'finger_millet': (25, 50, 20, 40, 20, 40, 20, 30, 50, 70, 5.5, 7.0, 80, 120),
    
    # Pulses
    'chickpea': (20, 50, 50, 80, 60, 90, 15, 25, 15, 35, 6.5, 8.0, 60, 100),
    'kidneybeans': (20, 45, 55, 80, 15, 30, 15, 25, 18, 25, 5.5, 7.0, 60, 120),
    'pigeonpeas': (10, 35, 50, 80, 15, 30, 20, 35, 30, 60, 5.0, 7.5, 100, 180),
    'mothbeans': (15, 35, 35, 60, 15, 30, 25, 35, 35, 55, 6.0, 8.0, 30, 70),
    'mungbean': (10, 35, 35, 60, 15, 30, 25, 35, 80, 95, 6.0, 8.0, 30, 70),
    'blackgram': (30, 50, 55, 75, 15, 30, 25, 35, 60, 75, 6.0, 8.0, 60, 100),
    'lentil': (10, 30, 55, 75, 15, 30, 18, 28, 30, 50, 6.0, 8.0, 40, 80),
    
    # Fruits
    'pomegranate': (10, 30, 10, 25, 30, 50, 20, 35, 35, 55, 5.5, 7.5, 35, 65),
    'banana': (90, 120, 70, 95, 45, 65, 23, 30, 75, 90, 5.5, 7.0, 100, 180),
    'mango': (15, 35, 15, 35, 25, 45, 25, 35, 45, 65, 5.5, 7.0, 90, 160),
    'grapes': (15, 35, 120, 150, 190, 210, 20, 35, 75, 90, 5.5, 7.0, 60, 90),
    'watermelon': (90, 110, 15, 30, 45, 60, 23, 30, 78, 90, 6.0, 7.0, 40, 60),
    'muskmelon': (90, 110, 15, 30, 45, 60, 26, 32, 88, 95, 6.0, 7.0, 20, 35),
    'apple': (15, 35, 120, 145, 195, 210, 20, 27, 88, 95, 5.5, 7.0, 100, 140),
    'orange': (15, 30, 10, 25, 5, 15, 20, 30, 88, 95, 6.5, 8.0, 100, 140),
    'papaya': (45, 65, 55, 75, 45, 60, 30, 42, 88, 95, 6.5, 7.5, 40, 60),
    'coconut': (15, 35, 10, 30, 25, 40, 25, 32, 93, 98, 5.5, 7.0, 150, 250),
    
    # Cash Crops
    'cotton': (110, 140, 35, 55, 15, 25, 22, 28, 75, 85, 5.8, 8.0, 50, 90),
    'jute': (70, 90, 35, 55, 35, 50, 23, 28, 75, 90, 6.0, 7.5, 150, 200),
    'sugarcane': (100, 150, 50, 80, 50, 80, 20, 35, 70, 85, 6.0, 8.0, 100, 200),
    'tobacco': (80, 120, 30, 60, 40, 80, 20, 30, 60, 80, 5.5, 7.5, 50, 100),
    
    # Oilseeds
    'groundnut': (20, 40, 40, 70, 25, 45, 25, 32, 60, 80, 5.5, 7.0, 50, 100),
    'sunflower': (60, 90, 25, 50, 30, 55, 20, 28, 50, 70, 6.0, 7.5, 50, 90),
    'mustard': (40, 70, 30, 55, 25, 45, 15, 25, 50, 70, 6.0, 8.0, 40, 80),
    'sesame': (25, 50, 20, 45, 20, 40, 25, 35, 40, 60, 5.5, 8.0, 40, 80),
    'castor': (20, 45, 20, 40, 15, 35, 20, 30, 50, 70, 5.0, 7.0, 40, 60),
    'soybean': (40, 70, 50, 80, 35, 55, 20, 30, 60, 80, 5.5, 7.0, 60, 120),
    
    # Spices
    'turmeric': (80, 120, 40, 70, 80, 120, 20, 30, 70, 90, 5.5, 7.5, 150, 250),
    'ginger': (70, 100, 50, 80, 70, 100, 22, 30, 80, 95, 5.5, 7.0, 200, 300),
    'cardamom': (75, 110, 75, 110, 75, 110, 15, 25, 85, 98, 5.0, 6.5, 250, 400),
    'pepper': (50, 80, 40, 70, 80, 120, 22, 30, 85, 95, 5.0, 6.5, 200, 350),
    'coriander': (30, 60, 40, 70, 30, 60, 20, 28, 50, 70, 6.5, 8.0, 40, 80),
    
    # Plantation Crops
    'arecanut': (60, 100, 40, 70, 80, 120, 22, 32, 80, 95, 5.0, 7.0, 200, 350),
    'cashew': (20, 40, 15, 35, 15, 35, 25, 35, 60, 80, 5.0, 7.0, 100, 200),
    'rubber': (40, 80, 20, 50, 30, 60, 25, 32, 80, 95, 4.5, 6.5, 200, 350),
    'tea': (40, 80, 20, 50, 30, 60, 18, 28, 80, 95, 4.5, 5.5, 200, 300),
    'coffee': (90, 110, 15, 30, 25, 40, 22, 28, 55, 70, 6.0, 7.0, 140, 180)
}

# Generate synthetic dataset
np.random.seed(42)
samples_per_crop = 150

data = []
for crop, params in crop_params.items():
    for _ in range(samples_per_crop):
        N = np.random.uniform(params[0], params[1])
        P = np.random.uniform(params[2], params[3])
        K = np.random.uniform(params[4], params[5])
        temperature = np.random.uniform(params[6], params[7])
        humidity = np.random.uniform(params[8], params[9])
        ph = np.random.uniform(params[10], params[11])
        rainfall = np.random.uniform(params[12], params[13])
        
        data.append([N, P, K, temperature, humidity, ph, rainfall, crop])

# Create DataFrame
df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save dataset
df.to_csv('Data/crop_recommendation.csv', index=False)
print(f"Dataset created with {len(df)} samples across {len(crop_params)} crops")
print(f"\nCrops included:")
for i, crop in enumerate(sorted(crop_params.keys()), 1):
    print(f"  {i:2d}. {crop}")

# Prepare features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model with more estimators for better accuracy
model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nTraining accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Save model
with open('models/RandomForest.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n[OK] Model saved to models/RandomForest.pkl")
print("You can now run the Flask app with: python app.py")
