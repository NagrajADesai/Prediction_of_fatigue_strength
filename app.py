import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, app, url_for
import pandas

app = Flask(__name__)
# load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    data =np.array(data).reshape(1,-1)
    print(data)
    output = regmodel.predict(data)[0]
    return render_template("home.html",prediction_text = "The life of given steel material is {} * 10^7 number of cycles".format(output))


if __name__ == "__main__":
    app.run(debug=True)
