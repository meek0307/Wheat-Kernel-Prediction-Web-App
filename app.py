from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)



## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            
            Area=request.form.get('Area'),
            Perimeter=request.form.get('Perimeter'),
            Compactness=request.form.get('Compactness'),
            Length_of_kernel=request.form.get('Length_of_kernel'),
            Width_of_kernel=request.form.get('Width_of_kernel'),
            Asymmetry_coefficient=(request.form.get('Asymmetry_coefficient')),
            Length_of_kernel_groove=(request.form.get('Length_of_kernel_groove'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        output = results
        if output == 1:
            return render_template("Kama.html")
        elif output == 2:
            return render_template("Rosa.html")
        else:
            return render_template("Canadian.html") 
    return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        
