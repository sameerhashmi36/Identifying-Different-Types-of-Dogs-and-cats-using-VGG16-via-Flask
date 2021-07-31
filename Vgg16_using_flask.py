#importing flask
from flask import Flask, render_template, request

import os

#importing Keras-Applications
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


app = Flask(__name__)
model = VGG16()

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    img_path = './images/' + imagefile.filename
    #print("----",imagefile.filename)
    imagefile.save(img_path)
    #print("1111111",img_path)

    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    predict_image = model.predict(image)
    label = decode_predictions(predict_image)
    label = label[0][0]

    # prediction
    classification = "%s (%.2f)" %(label[1], label[2]*100)
    #print("22222", label[1])
    
    # renaming the predicted image
    old_file_name = img_path
    new_file_name = "./images/" + label[1] + ".jpg"
    #print("33333333", new_file_name)
    os.rename(old_file_name, new_file_name)

    return render_template('index.html', prediction = classification)


if __name__ == '__main__':
    app.run(port=3000, debug=True)