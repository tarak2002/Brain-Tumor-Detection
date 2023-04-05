from flask import Flask,render_template,request
import tensorflow as tf
import numpy as np
import os
import urllib as ul
import uuid as ud
import cv2 as c
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = tf.keras.saving.load_model("mymodel.h5")


def predict1(imgname , model):
    img = tf.keras.utils.load_img(imgname , target_size = (150 , 150))
    img = tf.keras.utils.img_to_array(img)
    img = img/255
    img = np.expand_dims(img,axis = 0)
    result = model.predict(img)
    print(result)
    temp = np.argmax(result)

    return temp

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        filename = os.path.join(
            basepath, 'static/images', secure_filename(f.filename))
        f.save(filename)
        
        # Make prediction
        tumor_type = predict1(filename, model)
        
        if tumor_type==0:
            return  render_template('success.html' , img  = f.filename , predictions = "Glioma Tumor")
        elif tumor_type==1:
            return  render_template('success.html' , img  = f.filename , predictions = "Meningioma Tumor")
        elif tumor_type==2:
            return  render_template('success.html' , img  = f.filename , predictions = "No Tumor")
        elif tumor_type==3:
            return  render_template('success.html' , img  = f.filename , predictions = "Pituitary Tumor")
        else:
            return render_template('index.html')
        
if __name__ == "__main__":
    app.run(debug=True)


