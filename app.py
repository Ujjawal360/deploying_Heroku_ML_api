
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)

    
    if prediction == [0]:
        return render_template('index.html', prediction_text='You are not likely to have CHDs based on present condition')
    else:
        return render_template('index.html', prediction_text='Oopsie, You are likely to have CHDs based on present condition')

    


if __name__ == "__main__":
    app.run(debug=True)