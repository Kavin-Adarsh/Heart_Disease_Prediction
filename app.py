from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import keras 
from keras.models import Sequential 
from keras.layers import Dense 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split


data = pd.read_csv('heart.csv') 
X = data.iloc[:, [True, True, True,   True,     True,   True,  True,   True,  True,  False, False,  False, True,  False]].values 
dat = [[59, 1, 1, 140, 221, 0, 1, 164, 1, 2]] 
sc = StandardScaler() 
X_train = sc.fit_transform(X) 
model = load_model('models/model_X.h5')



app = Flask(__name__,  static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_request():
    data = request.get_json()
    request_text = data.get('request', '')

    print("\n\n")
    print("**************************************************************************")
    print("Request: ", request_text)
    

    dat = []

    try:
        for i in request_text:
            dat.append(int(i))

        print(dat)

        dat = [dat]
        
        dat = sc.transform(dat)
        pred = model.predict(dat)
        print(pred)
        pred = (pred > 0.5)

        response = ""

        if pred == False:
            response = "You don't have heart disease, and you will die alone"
            print("Response: You Have Heart Disease")

        elif pred == True:
            response = "You have heart disease, and you will die soon"
            print("Response: You dont' Have Heart Disease")

    except:
        response = "You probably don't have heart, if you have it, then enter all the values"

        print("Response: No Response")


    print("**************************************************************************")
    print("\n\n")



    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

