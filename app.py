from flask import Flask,render_template,redirect,request,send_file, session,send_from_directory
from tensorflow.keras.models import load_model
from flask_mysqldb import MySQL
import pickle
import numpy as np
import re
import os
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

model_file = "model.h5"
model = load_model(model_file)

app=Flask(__name__)
UPLOAD_FOLDER = 'prediction'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
#loading the pickle files of all 3 models which are used in read binary mode
filename = 'heart-disease-prediction-knn-model.pkl'
model1 = pickle.load(open(filename, 'rb'))
filename = 'covid19model.pkl'
clf = pickle.load(open(filename, 'rb'))

app.secret_key = 'yoursecretkey'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Abarna2528'
app.config['MYSQL_DB'] = 'Health'

mysql = MySQL(app)




@app.route('/')
def main():

    return render_template('index.html')

@app.route('/contactpage')
def cpage():
    return render_template('contact.html')


@app.route('/pa2', methods=["POST"])
def pa2():
    return render_template('page2.html')

@app.route('/pneuform')
def pneuform():
    return render_template('home.html')

@app.route('/signuppage')
def spage():
    return render_template('signup.html')
@app.route('/heart')
def heart():
    return render_template('main.html')
@app.route('/covidd')
def covidd():
    return render_template('covidd.html')

@app.route('/contact', methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        details = request.form
        first = details['first-name']
        last = details['last-name']
        email = details['email']
        mess = details['message']

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO contact(first, last, email, mess) VALUES (%s, %s, %s, %s)", (first, last, email, mess))
        mysql.connection.commit()
        cur.close()
        msg = 'Succesfully submitted !'

       
        
   
    return render_template('contact.html', msg=msg)

@app.route('/predict_2', methods=["GET", "POST"])
def predict_2():
    if request.method == "POST":
        myDict = request.form
        Fever = int(myDict['Fever'])
        Age = int(request.form['Age'])
        BodyPain = int(myDict['BodyPain'])
        RunnyNose = int(myDict['RunnyNose'])
        DiffBreath = int(myDict['DiffBreath'])
        ChestPain = int(myDict['ChestPain'])

        
        inputFeatures = np.array([[Fever, BodyPain, Age,
                         RunnyNose, DiffBreath, ChestPain]])
        InfProba = clf.predict(inputFeatures)
        #InfProba = int(InfProba)*100
        n='Not affected'
        if int(InfProba)>0:
            n='Affected'

        return render_template('show.html', inf=n)
    return render_template('covid.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        
        age = request.form['age']
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        my_prediction = model1.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

@app.route('/loginpage')
def lpage():
    return render_template('login.html')



@app.route('/signup', methods=['POST','GET'])
def signup():
    if request.method == "POST":
        details = request.form
        name = details['name']
        email = details['email']
        age = details['age']
        passw = details['passw']
        cur = mysql.connection.cursor()
       
        cur.execute('SELECT * FROM Register WHERE email = % s', (email, ))
        account = cur.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', name):
            msg = 'Username must contain only characters and numbers !'
       
        else:

            cur.execute("INSERT INTO Register(name, email, pass, age) VALUES (%s, %s, %s, %s)", (name, email, passw, age))
            mysql.connection.commit()
            cur.close()
            msg = 'You have successfully registered !'
    return render_template('signup.html', msg = msg)


@app.route('/pneu',methods=['GET','POST'])
def home():
    if request.method=='POST':
        if 'img' not in request.files:
            return render_template('home.html',filename="unnamed.png",message="Please upload an file")
        f = request.files['img'] 
        filename = secure_filename(f.filename) 
        if f.filename=='':
            return render_template('home.html',filename="unnamed.png",message="No file selected")
        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('home.html',filename="unnamed.png",message="please upload an image with .png or .jpg/.jpeg extension")
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if len(files)==1:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        else:
            files.remove("unnamed.png")
            file_ = files[0]
            os.remove(app.config['UPLOAD_FOLDER']+'/'+file_)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        predictions = makePredictions(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        return render_template('home.html',filename=f.filename,message=predictions,show=True)
    return render_template('home.html',filename='unnamed.png')

def makePredictions(path):
  '''
  Method to predict if the image uploaed is healthy or pneumonic
  '''
  img = Image.open(path) # we open the image
  img_d = img.resize((224,224))
  # we resize the image for the model
  rgbimg=None
  #We check if image is RGB or not
  if len(np.array(img_d).shape)<3:
    rgbimg = Image.new("RGB", img_d.size)
    rgbimg.paste(img_d)
  else:
      rgbimg = img_d
  rgbimg = np.array(rgbimg,dtype=np.float64)
  rgbimg = rgbimg.reshape((1,224,224,3))
  predictions = model.predict(rgbimg)
  a = int(np.argmax(predictions))
  if a==1:
    a = "pneumonic"
  else:
    a="healthy"
  return a

@app.route('/login',methods=['POST','GET'])
def login():
    msg = ''
    if request.method == 'POST':
        email = request.form['email']
        passw = request.form['pass']
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM Register WHERE email = % s AND pass = % s', (email, passw, ))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
         
            session['name'] = account[0]
        
            return render_template('page2.html')
        else:
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg = msg)
            
    




@app.route('/logout')
def logout():

    session.pop('loggedin', None)

    session.pop('name', None)
    return render_template('index.html')



if __name__=='__main__':
    app.run(debug=True)