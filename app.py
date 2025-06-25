import pandas as pd
import pickle
import xgboost as xgb
import numpy as np

# ---------------------------
# 1. Load the Saved Artifacts
# ---------------------------
# Load the XGBoost model (saved as JSON)
model = xgb.XGBRegressor()
model.load_model('student_performance_xgb_improved.json')

# Load the StandardScaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the LabelEncoders for each categorical feature
with open('le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)
with open('le_race_ethnicity.pkl', 'rb') as f:
    le_race = pickle.load(f)
with open('le_parental_level_of_education.pkl', 'rb') as f:
    le_parent_edu = pickle.load(f)
with open('le_lunch.pkl', 'rb') as f:
    le_lunch = pickle.load(f)
with open('le_test_preparation_course.pkl', 'rb') as f:
    le_test_prep = pickle.load(f)

# ---------------------------
# 2. Define Preprocessing Function
# ---------------------------
def preprocess_input(data_dict):
    """
    Preprocess a dictionary of input data so that it can be fed to the model.
    Expects keys:
      - 'gender'
      - 'race/ethnicity'
      - 'parental level of education'
      - 'lunch'
      - 'test preparation course'
      - 'reading score'
      - 'writing score'
      
    Note: math score is not expected as it's the target.
    """
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Transform categorical features using the loaded LabelEncoders
    df['gender'] = le_gender.transform(df['gender'])
    df['race/ethnicity'] = le_race.transform(df['race/ethnicity'])
    df['parental level of education'] = le_parent_edu.transform(df['parental level of education'])
    df['lunch'] = le_lunch.transform(df['lunch'])
    df['test preparation course'] = le_test_prep.transform(df['test preparation course'])
    
    # Feature engineering: create interaction and aggregate features
    df['parent_edu_lunch'] = df['parental level of education'] * df['lunch']
    df['prep_parent_edu'] = df['test preparation course'] * df['parental level of education']
    df['avg_reading_writing'] = (df['reading score'] + df['writing score']) / 2
    # Since math score is unknown during prediction, calculate standard deviation over reading and writing only
    df['score_std'] = df[['reading score', 'writing score']].std(axis=1)
    
    # Ensure the features are in the same order as used during training
    feature_cols = [
        'gender', 
        'race/ethnicity', 
        'parental level of education', 
        'lunch', 
        'test preparation course',
        'parent_edu_lunch', 
        'prep_parent_edu', 
        'avg_reading_writing', 
        'score_std'
    ]
    
    # Scale the features using the loaded scaler
    X = scaler.transform(df[feature_cols])
    return X

# ---------------------------
# 3. Make a Prediction
# ---------------------------
# Define a sample input (adjust values as needed)
sample_input = {
    "gender": "female",
    "race/ethnicity": "group D",
    "parental level of education": "some college",
    "lunch": "standard",
    "test preparation course": "completed",
    "reading score": 70,
    "writing score": 100
}

# Preprocess the input
X_new = preprocess_input(sample_input)

# Predict the math score using the loaded model
prediction = model.predict(X_new)

print("Predicted math score:", prediction[0])
