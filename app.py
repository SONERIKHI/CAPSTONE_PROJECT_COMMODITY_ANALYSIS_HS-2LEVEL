from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

filename = 'RandomForestRegressor.pkl'
classifier = pickle.load(open(filename, 'rb'))
model = pickle.load(open('RandomForestRegressor.pkl','rb'))

app = Flask(__name__, template_folder="templates")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_value():
    try:
        Export_Value = int(request.form['Export_Value'])
        Import_Value = int(request.form['Import_Value'])
        
        input_features = [Export_Value,Import_Value]
        features_value = [np.array(input_features)]
        feature_name = ['Export_Value', 'Import_Value']

        df = pd.DataFrame(features_value, columns=feature_name)
        output = classifier.predict(df)

        return render_template('index.html', prediction_text='calculate : {:.2f}'.format(output[0]))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

if __name__ == "__main__":
    app.run(debug=True)
