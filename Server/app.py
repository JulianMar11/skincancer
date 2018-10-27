from flask import Flask, render_template, request
import os
app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(APP_ROOT,'static/')

def prediction():
    return "Dies ist ein Schloss"

@app.route("/")
def prediction():
    return "Dies ist ein Schloss"

@app.route("/predictImage", methods=['POST'])
def predictImage():
	#print("Hallo")
    for file in request.files.getlist("image"):
        direction = '/'.join([imageDir,file.filename])
        file.save(direction)
     #   filename = file.filename
      #  prediction1 = prediction()
    return "Hallo"



if __name__ == '__main__':
   app.run(host='0.0.0.0', port=80)
