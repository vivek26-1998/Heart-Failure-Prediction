from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Assuming you have defined numerical_cols and categorical_cols as before
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

@app.route("/")
def home():
    return render_template("model.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Collect form data
        age = int(request.form['age'])
        sex = request.form['sex']
        chest_pain_type = request.form['chest_pain_type']
        resting_bp = int(request.form['resting_bp'])
        cholesterol = int(request.form['cholesterol'])
        fasting_bs = int(request.form['fasting_bs'])
        resting_ECG = request.form['resting_ECG']
        maxHR = int(request.form['maxHR'])
        exercise_angina = request.form['exercise_angina']
        old_peak = float(request.form['old_peak'])
        ST_slope = request.form['ST_slope']

        # Create a DataFrame from the input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ChestPainType': [chest_pain_type],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'RestingECG': [resting_ECG],
            'MaxHR': [maxHR],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak': [old_peak],
            'ST_Slope': [ST_slope]
        })

        # Ensure categorical columns are of type 'object'
        for col in categorical_cols:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype('object')

        # Perform prediction
        prediction = model.predict(input_data)

        # Handle prediction result
        if prediction[0] == 0:
            return render_template("negative.html")
        else:
            return render_template("positive.html")

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Route for processing uploaded CSV file
@app.route("/upload_csv", methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        try:
            df = pd.read_csv(file)
            
            # Drop 'HeartDisease' column if it exists
            if 'HeartDisease' in df.columns:
                df.drop('HeartDisease', axis=1, inplace=True)
            
            # Ensure DataFrame columns match expected input for model
            X = df[numerical_cols + categorical_cols]
            
            # Perform predictions
            predictions = model.predict(X)
            
            # Convert predictions to meaningful labels
            prediction_labels = ['YES' if pred == 1 else 'NO' for pred in predictions]
            
            df['HeartFailure'] = prediction_labels
            
            # Render results.html with the DataFrame as a table
            return render_template("result.html", tables=[df.to_html(classes='data', header=True, index=False)])
        
        except pd.errors.EmptyDataError:
            return "Empty file uploaded. Please upload a valid CSV file."
        
        except Exception as e:
            return f"An error occurred: {str(e)}"

    return "Unknown error occurred"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)