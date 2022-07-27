import YOLO as yl
import os
from flask import Flask, render_template, request, redirect, url_for
PEOPLE_FOLDER = os.path.join('', 'display')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

# @app.route('/display/<filename>')
# def display_image(filename):

#     return redirect(url_for('display_image',filename='object-detection.jpg'), code=301)


@app.route('/', methods=['POST'])
def getFilePath():
    imageFile = request.files['imagefile']
    imagePath = "./images/" + imageFile.filename
    imageFile.save(imagePath)
    filename = yl.getDetect(imagePath)
    return render_template('index.html', filename=filename)


@app.route('/display/<filename>')
def display_image(filename):

    return redirect(filename=('../display/'+filename), code=301)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
