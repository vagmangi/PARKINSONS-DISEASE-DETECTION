import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('testing.html')

@app.route('/predict',methods=['POST','GET'])
def predict():

    int_features = [float(x) for x in request.form.values()] 
    features = [np.array(int_features)]  
    prediction = model.predict_proba(features)  

    output = round(prediction[0], 2)

    return render_template('testing.html', prediction_text="Parkinson's Disease {}".format(output))

if __name__ == "__main__":
    app.run()