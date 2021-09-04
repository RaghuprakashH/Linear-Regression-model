import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from tutor_model import train_model
from tutor_model import Predict_model




app = Flask(__name__)
model = pickle.load(open('tutor_model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''

        For rendering results on HTML GUI
    '''
    #train_model()

    #model = pickle.load(open('tutor_model.pkl', 'rb'))
    features = [x for x in request.form.values()]
    int_features =[]

    for count, i in enumerate(features):
        if count == 0:
            int_features.append(float(i))
        if count == 1:
            int_features.append(int(i))
        if count == 2:
            int_features.append(float(i))
        if count == 3:
            int_features.append(int(i))
        if count == 4:
            int_features.append(int(i))
        if count == 5:
            int_features.append(int(i))
        if count == 6:
            int_features.append(int(i))
        if count == 7:
            int_features.append(int(i))
        if count == 8:
            int_features.append(int(i))


    #int_features = [F1,I1,F2,I2,I3,I4,I5,I6,I7,I8]
    #final_features = [np.array(int_features)]
    #features1 = pd.DataFrame(features)
    #model(int_features)
    #a = Predict_model(int_features)
    #b= []
    #for i in a.new_features:
    #    b.append(i)
    #scaled_val=Predict_model.preprocess_predict(b)
    X1 = pd.read_pickle('t_data.pkl')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit_transform(X1)
    scaled_val = scaler.transform([int_features])
    #scale_final_features = a.preprocess_predict(features1)
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #final_features1 = scaler.transform(final_features)
    #features = scaler.transform(np.array([[202.6, 1433, 39.5, 7, 0, 0, 0, 0, 0]]))
    prediction = model.predict(scaled_val)
    #accuracy = Predict_model.adj_r2(scaled_val,prediction)

    #r2 = model.score(scaled_val,prediction)
    #print(r2)

    return render_template('index.html', prediction_text='Air Temperature should be {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)