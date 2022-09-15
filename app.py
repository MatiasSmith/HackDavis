

from flask import Flask, redirect, url_for, render_template, request, session, flash, send_file
from datetime import timedelta
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from flask import Flask, redirect, url_for, render_template, request, session, flash, send_file
from datetime import timedelta
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os 
import tensorflow
import keras
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential
import numpy as np
import image_dataset_loader
import csv


#from views import views
#import numpy as np
#import pytorch as py


# This is a test to see if I can commit properly

app = Flask(__name__)
#app.register_blueprint(views, url_prefix='/views')
#secret key: way that we decrypt and encrypt data


app.config['SECRET_KEY'] = 'bobross'
app.config['UPLOAD_FOLDER'] = 'PneumoniaDataset/UserInput/files'
app.config['UPLOAD_FOLDER2'] = 'PneumoniaDataset/TrainingData'
app.secret_key = "bobross"
app.permanent_session_lifetime = timedelta(days=5)


def pneumoniaOperator():
    #Load a dataset of images, ignoring their labelling, from a single folder a directory below UserInput. Due to requirements, a second folder is required there but should always be left empty. All of the files are collected in default operating system order.
    (dataImage, unusedTrainDataLabel), = image_dataset_loader.load('./PneumoniaDataset', ['UserInput'], shuffle=False)

    print(len(dataImage))

    model = keras.models.load_model('PneumoniaModel.h5')
    imageNum = len(os.listdir('./PneumoniaDataset/UserInput/files'))

    np.reshape(dataImage, (imageNum, 64, 64, 3))



    output = model.predict(dataImage)


    #Program output is via csv files, with the first column for names and the second for diagnosis.
    #Ensure the that "ignore" folder is empty or else the program will crash from patientNames going OoB
    file = open('prediction.csv', 'w')
    writer = csv.writer(file)
    patientNames = os.listdir('./PneumoniaDataset/UserInput/files')
    patients = np.array(patientNames)


    count = 0
    for i in output:
        #i contains the predictions, and its second element show's the model's Pneumonia Likelihood
        if (i[1] > 0.00367):
            writer.writerow([patients[count], 'Pneumonia'])
        else:
            writer.writerow([patients[count], 'Healthy'])
        count = count + 1

    file.close()
    return redirect(url_for("download_file")) 

def train():
    #This special function imports training/testing images/labels from a hierarchy of folders, inferring labels based on folder names and setting based on similar metadata between images.
    (trainDataImage, trainDataLabel), = image_dataset_loader.load('./PneumoniaDataset', ['TrainingData'])

    model = Sequential()

    model.add(Conv2D(60, kernel_size = 1, activation='relu', input_shape=(64, 64, 3), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(35, kernel_size = 1, activation='relu', padding='same'))
    model.add(Dropout(0.2))

    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(20, kernel_size = 1, activation='relu', padding='same'))
    model.add(Dropout(0.2))

    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(10, kernel_size = 1, activation='relu', padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(2, activation='softmax'))

    model.summary()

    # This function will take a variable, number of rows (corresponds to the number of images/labels in the variable), and check for a boolean that shows whether the variable being reshaped is an image
    def Reshaper(var, imNumber, isImage):
        if(isImage==True):
            np.reshape(var, (imNumber, 64, 64, 3))
        else:
            np.reshape(var, (imNumber, 1))
        return var

    trainDataImage = Reshaper(trainDataImage, 2800, True)
    trainDataLabel = Reshaper(trainDataLabel, 2800, False)

    def OneHotEncode(DataLabel, labelNum):
        OneHot = np.zeros((labelNum, 2))
        count=0

        while (count < labelNum):
            OneHot[count][DataLabel[count].astype(int)] = 1
            count=count+1

        return OneHot

    trainDataLabel = OneHotEncode(trainDataLabel, 2800)



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Here, we give the model a training dataset to estimate from and a training answer set to score itself on, which correspond to the images and labels respectively.
    model.fit(trainDataImage, trainDataLabel, epochs=25, batch_size=25)

    model.save('PneumoniaModel2.h5')

#Webpage #1
@app.route("/", methods=['GET', "POST"])
def home():
    if request.method == 'POST':
        
        if request.form.get("GetModel") == "MODEL":
            train()
            return redirect(url_for("home")) 
        
        elif request.form.get("GetResults") == "RESULTS":
            pneumoniaOperator()
            return redirect(url_for("home"))

        
        elif request.form.get("About") == "ABOUT":
            return redirect(url_for("about"))
        
        # elif request.form.get("Home") == "HOME":
        #     about = request.form["Home"]
        #     session["home"] = about
        #     return redirect(url_for("logout"))
        
        
        elif request.form.get("Team") == "TEAM":
            return redirect(url_for("team"))

        
        elif request.files.getlist('files'):
            
            for f in request.files.getlist('files'):
                if f.filename:
                    f.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
                else:
                    return render_template("home.html")
        
        elif request.files.getlist('files2'):
            for f in request.files.getlist('files2'):
                if f.filename:
                    f.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER2'], secure_filename(f.filename)))
                else:
                    return render_template("home.html")


        else:
            return render_template("home.html")
    return render_template("home.html")
    
    
#For Downloading a file
@app.route('/download')
def download_file():
    p = "prediction.csv"
    return send_file(p, as_attachment=True)

        
        # get_request(request.form.get('action1'), request.form.get("About"), request.form.get("Home"), request.form.get("Team"), request.form.get("Settings"))
        
        


        #Set our session to be valid for a certain amount of time
        #even if you close webpage
        # session.permanent = True
        # user = request.form["nm"]
        # session["user"] = user
        # session["user2"] = 12
        # flash("Login Successful")
        #return redirect(url_for("user", usr=user, usr2=user2))
        # return redirect(url_for("user"))
    


    
@app.route("/about", methods=["POST", "GET"])
def about():

    if request.method == "POST":
        
        if request.form.get("About") == "ABOUT":
            about = request.form["About"]
            session["about"] = about
            return redirect(url_for("about"))
        
        elif request.form.get("Home") == "HOME":
            about = request.form["Home"]
            session["home"] = about
            return redirect(url_for("/"))
        
        
        elif request.form.get("Team") == "TEAM":
            about = request.form["Team"]
            session["team"] = about
            return redirect(url_for("team"))


    else:
        #If user has already logged in and is in session
        if "user" in session:
            flash("Already Logged in!")
            return redirect(url_for("user"))
        return render_template("about.html", name="About")
    



@app.route("/team", methods=["POST", "GET"])
def team():

    if request.method == "POST":
        
        if request.form.get("About") == "ABOUT":
            about = request.form["About"]
            session["about"] = about
            return redirect(url_for("about"))
        
        elif request.form.get("Home") == "HOME":
            about = request.form["Home"]
            session["home"] = about
            return redirect(url_for("/"))
        
        
        elif request.form.get("Team") == "TEAM":
            about = request.form["Team"]
            session["team"] = about
            return redirect(url_for("team"))

    else:
        #If user has already logged in and is in session
        if "user" in session:
            flash("Already Logged in!")
            return redirect(url_for("user"))
        return render_template("team.html", name="Team")


if __name__ == '__main__':
    app.run(debug=True)
    

