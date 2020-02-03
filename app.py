from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from datetime import datetime
import time,timeit
import pdb

from skimage import data, img_as_float
from skimage import exposure

# Define a flask app
app = Flask(__name__)
app.config['UPLOAD_PATH'] = "/home/arun/retinopathy/keras-flask-deploy-webapp/uploads"

# Model saved with Keras model.save()
MODEL_PATH = '/media/arun/data/wellcare2/data-aug-simple-work/models/bestmodel.h5'

# Load your trained model
print("Starting to load model...",MODEL_PATH)
start_time = timeit.default_timer()
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

elapsed = timeit.default_timer() - start_time
print("Time to load model = ",elapsed)

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
print('Ready! Check http://127.0.0.1:5000/')


def img_preprocess(x):

    p2, p98 = np.percentile(x, (2, 98))
    x = exposure.rescale_intensity(x, in_range=(p2, p98))
    #x = exposure.equalize_adapthist(x, clip_limit=0.03)
    x = exposure.equalize_hist(x)
    return x


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = img_preprocess(x)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #test_datagen = ImageDataGenerator(rescale=1./255)

    #x = preprocess_input(x, mode='torch')
    x = x / 255.

    preds = model.predict(x,verbose=0)
    #pdb.set_trace()
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/multi', methods=['GET'])
def multi():
    # Main page
    return render_template('multiple.html')


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



        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #red_class = decode_predictions(preds, top=1)   # ImageNet Decode

        #y_pred = np.argmax(preds, axis=1)
        # preds is a 1x2 array with probabilities of each predicted class on axis=1
        # the following is the index of the highest prob prediction
        y_pred = np.argmax(preds)
        # this is the probability of the highest prediction
        y_prob = preds[0][y_pred]
        #print("Entering debug mode - after predictions")
        #pdb.set_trace()

        class_names =  {0: 'nongradable', 1:'normal',  2:'retinopathy'}

        result = class_names[y_pred]                          # this is the name of the class predicted
        y_prob   = '; Probabilty = {:.2%}'.format(y_prob)     # this is the probability
        result +=  y_prob

        return result
    return None


@app.route('/predictmulti', methods=['GET', 'POST'])
def uploadmulti():
    if request.method == 'POST':
        # Get the file from post request
        results = ""
        class_names =  {0: 'nongradable', 1:'normal',  2:'retinopathy'}
        for f in request.files.getlist('file'):
            file_path = os.path.join(app.config['UPLOAD_PATH'], secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            preds = model_predict(file_path, model)

            #y_pred = np.argmax(preds, axis=1)
            # preds is a 1x2 array with probabilities of each predicted class on axis=1
            # the following is the index of the highest prob prediction

            y_pred = np.argmax(preds)

            # this is the probability of the highest prediction
            y_prob = preds[0][y_pred]

            #print("Entering debug mode - after predictions")
            #pdb.set_trace()


            result = class_names[y_pred]               # Convert to string
            y_prob   = '; Probabilty = {:.2%}'.format(y_prob)
            result +=  y_prob + "\n\n"
            results +=  result

        return results
    return None



if __name__ == '__main__':
    # app.run(port=5000, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
