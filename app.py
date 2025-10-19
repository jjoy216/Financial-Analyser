# create flask app
# import the necessary packages 
from flask import Flask, render_template,jsonify, request    
import pickle
import numpy as np          

# create the flask app
app = Flask(__name__)

# load the models vectorizers
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
# load vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vec = pickle.load(f)

# define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # get the message from the form input
    message = request.form['message']
    data = [message]
    vect = vec.transform(data).toarray()
    my_prediction = model.predict(vect)
    # normalize the prediction so it can be JSON-serialized safely
    pred = my_prediction[0]
    # if the model returned an array-like element (e.g., [[label]]), unwrap it
    if isinstance(pred, (list, tuple, np.ndarray)):
        try:
            if len(pred) > 0:
                pred = pred[0]
        except Exception:
            pass

    # convert numpy scalar to native python type
    try:
        if isinstance(pred, np.generic):
            pred = pred.item()
    except Exception:
        pass

    # finally, try to convert to int when appropriate, otherwise return string
    try:
        pred_out = int(pred)
    except Exception:
        pred_out = str(pred)

    return jsonify({'prediction': pred_out})

if __name__ == '__main__':
    app.run(debug=True)