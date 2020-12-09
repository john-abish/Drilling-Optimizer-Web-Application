import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('RFmodel.pkl', 'rb'))
scaler = pickle.load(open('Scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', redirect_text='Access source code on GITHUB')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    '''
    int_input = [int(x) for x in request.form.values()]
    final_input = [np.array(int_input)]
    scaled_input = scaler.transform(final_input)
    prediction = model.predict(scaled_input)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The ROP is predicted to be: {} ft/h'.format(output), redirect_text='Access source code on GITHUB')




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    #app.run(debug=True)
