import joblib
import numpy as np
import math

from flask import Flask,jsonify,render_template,request

app = Flask(__name__,template_folder='../template',static_folder='../template/css')

#POST para pruebas
clf = joblib.load('../models/diabetes_model.pkl')

@app.route('/', methods=['GET','POST'])
def prediction():
    #array de testeo
    #X= np.array([6,148,72,35,0,33.6,0.627,50])
    
    #toma de datos
    if request.method == 'POST':
        embarazos = float(request.form.get('embarazos'))
        glucosa = float(request.form.get('glocusa'))
        pre_sangre = float(request.form.get('pre_sangre'))
        gro_piel = float(request.form.get('gro_piel'))
        insulina = float(request.form.get('insulina'))
        grasa_corp = float(request.form.get('grasa_corp'))
        hist_fam = float(request.form.get('hist_fam'))
        edad = float(request.form.get('edad'))
        
    
        X = np.array([embarazos,glucosa,pre_sangre,gro_piel,insulina,grasa_corp,hist_fam,edad])
    
    
        predict_model = round(clf.predict(X.reshape(1,-1)).tolist()[0][0])
        print(predict_model)
    else:
        predict_model = ""
        
    #return jsonify({'prediction': list(predict_model)})
    return render_template('index.html',output=predict_model)
    
app.run(port=8000)  