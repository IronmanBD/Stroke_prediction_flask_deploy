import numpy as np
from flask import Flask, request,  render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    gender = int(request.form.get("gender"))
    age = int(request.form.get("age"))
    hypertension = int(request.form.get("hypertension"))
    heart_disease = int(request.form.get("heart_disease"))
    ever_married = int(request.form.get("ever_married"))
    work_type = int(request.form.get("work_type"))
    residence_type = int(request.form.get("Residence_type"))
    glucose_level = float(request.form.get("avg_glucose_level"))
    bmi = float(request.form.get("bmi"))
    smoking_status = int(request.form.get("smoking_status"))
    
    int_features = [gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, glucose_level, bmi, smoking_status]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('index.html',pred='Stroke Probability is {}'.format(output))
    else:
        return render_template('index.html',pred='Stroke Probability is {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)