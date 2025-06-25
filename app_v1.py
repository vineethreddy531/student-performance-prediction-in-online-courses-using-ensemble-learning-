from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import xgboost as xgb

app = Flask(__name__)


model = xgb.XGBRegressor()
model.load_model('student_performance_xgb_improved.json')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

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


def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    
    df['gender'] = le_gender.transform(df['gender'])
    df['race/ethnicity'] = le_race.transform(df['race/ethnicity'])
    df['parental level of education'] = le_parent_edu.transform(df['parental level of education'])
    df['lunch'] = le_lunch.transform(df['lunch'])
    df['test preparation course'] = le_test_prep.transform(df['test preparation course'])
    
    df['parent_edu_lunch'] = df['parental level of education'] * df['lunch']
    df['prep_parent_edu'] = df['test preparation course'] * df['parental level of education']
    df['avg_reading_writing'] = (df['reading score'] + df['writing score']) / 2
    df['score_std'] = df[['reading score', 'writing score']].std(axis=1)
    
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
    X = scaler.transform(df[feature_cols])
    return X


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        X = preprocess_input(data)
        prediction = model.predict(X)
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
