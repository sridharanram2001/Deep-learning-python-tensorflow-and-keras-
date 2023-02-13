from flask import Flask,render_template,session,url_for,redirect
import numpy as np
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from tensorflow.keras.models import load_model
import joblib


def return_prediction(model,scaler,sample_json):
    
    
    s_len = sample_json["sepal_length"]
    s_wid = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_wid = sample_json["petal_width"]
    
    flower = [[s_len,s_wid,p_len,p_wid]]
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    flower = scaler.transform(flower)
    
    class_ind = model.predict_classes(flower)
    
    return classes[class_ind][0]



app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class FlowerForm(FlashForm):

	sep_len = TextField("Sepal Length")
	sep_wid = TextField("Sepal Width")
	pet_len = TextField("Petal Length")
	pet_wid = TextField("Petal Width")
	submit = SubmitField("Analyse")





@app.route("/",methods =['GET','POST'])
def index():

	form = FlowerForm()

	if form.validate_on_submit():

		session['sep_len'] = form.sep_len.data
		session['sep_wid'] = form.sep_wid.data
		session['pet_len'] = form.pet_len.data
		session['pet_wid'] = form.pet_wid.data

		return redirect(url_for("prediction"))
	return render_template('home.html',form=form)
	

flower_model = load_model('final_iris_model1.h5')
flower_scaler = joblib.load('iris_scaler1.pkl')

@app.route('/prediction')
def prediction():
	contents ={}

	contents['sepal_length'] = float(session['sep_len'])
	contents['sepal_width'] = float(session['sep_wid'])
	contents['petal_length'] = float(session['pet_len'])
	contents['petal_width'] = float(session['pet_wid'])

	results = return_prediction(flower_model,flower_scaler,contents)

	return render_template('prediction.html',results=results)
if __name__=='__main__':
	app.run()
