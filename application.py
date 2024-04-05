from flask import Flask, request, app, render_template
from flask import request
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

with open("Model/scaler.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open("Model/ModelForPrediction.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

# @app.route('/')
# def index():
#     return render_template('index.html')

# route for single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result = ""
    
    if(request.method == 'POST'):
        
        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get("Glucose"))
        BloodPressure = float(request.form.get("BloodPressure"))
        SkinThickness = float(request.form.get("SkinThickness"))
        Insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("Pregnancies"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = float(request.form.get("Age"))
        
        new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = model.predict(new_data)

        if prediction[0]==1:
            result = 'DIABETIC'
        else:
            result = 'NON-DIABETIC'
            
        return render_template('single_prediction.html', result= result)
    else:
        return render_template('home.html')        


if __name__ == "__main__":
    app.run(host="0.0.0.0")
        








