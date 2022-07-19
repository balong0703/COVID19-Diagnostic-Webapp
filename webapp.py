from flask import Flask,render_template,url_for,request,redirect,flash
import os
import urllib
import torch
from torch_utils import Net_Effb0,Net_CNN,Prediction

app = Flask(__name__)

allow_ext = ['jpg','jpeg']
classes = ["COVID-19","Normal","Pneumonia-Bacterial","Pneumonia-Viral"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load model
model = Net_Effb0()
checkpoint = torch.load("checkpoint_effb0_3.pt",map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(device)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/home")
def homepage():
    return render_template("index.html")

@app.route("/predict",methods=["POST","GET"])    
def predict():
    error = ""
    if request.method == "POST":
        img = request.files["my_image"]
        if (img.filename.rsplit('.')[-1].lower() not in allow_ext):
            error = "Vui lòng tải lên ảnh .jpg hoặc .jpeg"
        else:
            img_path = 'static/image_upload/'+img.filename
            img.save(img_path)

            #predict image
            index = Prediction(img_path, model)

        if (len(error) == 0):
            return render_template("predict.html", img_path = img_path, label = classes[index])
        else:
            return render_template("predict.html", error = error)
 
    else:
        return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)