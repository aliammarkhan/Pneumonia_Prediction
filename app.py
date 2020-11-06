# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:44:21 2020

@author: Biohazard
"""
from __future__ import division, print_function
# coding=utf-8

import os
import numpy as np
#import tensorflow as tf
# Keras
#from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image
import cv2

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='my_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    #load image
    img = cv2.imread(img_path)
    #rescale image
    img = img*(1/255.0)
    #resizing image
    img = cv2.resize(img, (224,224))
    #expand dims
    img = np.expand_dims(img, axis=0)
    #predict
    preds = model.predict(img)
    
    
    if preds[0] >0.5:
        preds = ':( It looks like you probably have Pneumonia'
    else:
        preds = 'Dont Worry you are just fine :)'
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)