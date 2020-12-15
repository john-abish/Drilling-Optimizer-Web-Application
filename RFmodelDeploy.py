import json
import numpy as np
from flask import Flask, request, render_template, send_file, Response
import pickle
import io
from pyswarms.single.global_best import GlobalBestPSO
from io import StringIO
import os
import time


app = Flask(__name__, static_url_path='', static_folder='templates', template_folder='templates')
model = pickle.load(open('RFmodel.pkl', 'rb'))
scaler = pickle.load(open('Scaler.pkl', 'rb'))

@app.route('/', methods=['POST', 'GET'])
def base():
    return render_template('index.html')

@app.route('/home',methods=['POST', 'GET'])
def home():
    return render_template('index.html')

@app.route('/predict_page',methods=['POST', 'GET'])
def predict_page():
    return render_template('index-predict.html')

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    '''
    '''
    int_input = [int(float(x)) for x in request.form.values()]
    final_input = [np.array(int_input)]
    scaled_input = scaler.transform(final_input)
    prediction = model.predict(scaled_input)

    output = round(prediction[0], 2)

    return render_template('index-predict.html', const_text='The ROP is predicted to be: ', prediction_text= '{} ft/h'.format(output))

@app.route('/chartdata')
def chartdata():
    
    def generate_chart():
    
        x_max_list = [[17000, 145, 1180]]
        x_min_list = [[10000,100,1170]]
        x_max_scaled=scaler.transform(np.array(x_max_list))
        x_min_scaled=scaler.transform(np.array(x_min_list))
        bounds = (x_min_scaled[0], x_max_scaled[0])
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        Opt_ROP=[]
        Opt_para=[]
        del_h=[]
        y_i1=0
        Wf=1
        for _ in range(5340,5941,10):
            y_i2=np.cbrt(((_*1.87*10**-10)/(1-y_i1))+(y_i1**3))
            y_i1=y_i2
            del_h.append(y_i2)
            def testROP(x):
                f = model.predict(x)
                return -Wf*f
            optimizer = GlobalBestPSO(n_particles=50, dimensions=3, options=options, bounds=bounds)
            cost, pos = optimizer.optimize(testROP, 2)
            
            json_data = json.dumps(
                {'optROP': -1 * cost, 'value': _})
            yield f"data:{json_data}\n\n"
            time.sleep(0.05)
            
            Opt_ROP.append(cost)
            Opt_para.append(pos)
            Wf=1-y_i2
        #neg_ROP=[i*-1 for i in Opt_ROP] 
        
        #fig = Figure()
        #axis = fig.add_subplot(1, 1, 1)
        #xs = range(0,601,10)
        #ys = neg_ROP
        #axis.plot(xs, ys)
        #return fig 
        #plt.plot(range(0,601,10), neg_ROP)
        #plt.xlabel("Relative Depth (ft)")
        #plt.ylabel("Optimal ROP (ft/h)")
        #if 'plot.jpg' in os.listdir('templates/'):
        #    os.remove('templates/plot.jpg')
        #plt.savefig('templates/plot.jpg')
        time.sleep(86400)
        
    return Response(generate_chart(), mimetype='text/event-stream')

    #def ROPplot():
    #    fig = optimize()
    #    output = io.BytesIO()
    #    FigureCanvas(fig).print_png(output)
    #    return Response(output.getvalue(), mimetype='image/png')

@app.route('/optimize',methods=['POST', 'GET'])
def optimize():
    return render_template('index-chart.html')

@app.route('/volve',methods=['POST', 'GET'])
def volve():
    return render_template('index-volve.html')

@app.route('/pso',methods=['POST', 'GET'])
def pso():
    return render_template('index-pso.html')

@app.route('/rf',methods=['POST', 'GET'])
def rf():
    return render_template('index-rf.html')

@app.route('/about',methods=['POST', 'GET'])
def about():
    return render_template('index-about.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
 