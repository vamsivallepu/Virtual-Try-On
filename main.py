from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from Model import Model
import matplotlib.pyplot as plt
import time
import cv2

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = Model("checkpoints/jpp.pb",
              "checkpoints/gmm.pth",
              "checkpoints/tom.pth")


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded images
        person_image = request.files['person']
        cloth_image = Image.open(request.form['cloth'][22:])
        print(type(person_image), type(cloth_image))

        # Save the uploaded images to the server
        person_filename = 'person.jpg'
        cloth_filename = 'cloth.jpg'
        person_image.save(person_filename)
        cloth_image.save(cloth_filename)

        # Open the saved images and preprocess them for the model
        person_image = np.array(Image.open(person_filename))
        cloth_image = np.array(Image.open(cloth_filename))

        # Use the model to generate the output image
        start = time.time()
        result, trusts = model.predict(person_image, cloth_image, need_pre=False, check_dirty=True)
        if result is not None:
            end = time.time()
            print("time:"+str(end-start))
            print("Confidence:"+str(trusts))
            result=cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite('static/result.jpg', result)

        # Display the output image on the website
        return render_template('result.html')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
