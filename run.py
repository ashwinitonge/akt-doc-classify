"""
Mortgage document classifier
Created by Ashwini Tonge
"""
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, request,flash, jsonify, render_template, make_response
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.models import load_model
from flask_swagger_ui import get_swaggerui_blueprint


DEBUG = True
app = Flask(__name__)

SWAGGER_URL = '/api/docs' # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger.json' # Our API url (can of course be a local resource)

swaggerui_blueprint = get_swaggerui_blueprint(
SWAGGER_URL, # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
API_URL,
config={ # Swagger UI config overrides
'app_name': "Mortgage Document Classification"
}
)

# Register blueprint at URL
# (URL must match the one given to factory function above)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
model = load_model('Sequence_model/lstm.h5')
token = joblib.load('Sequence_model/token.pkl')

#app.clf = joblib.load('Sequence_model/lstm.pkl')
#app.token = joblib.load('Sequence_model/token.pkl')

class ReusableForm(Form):
    words = TextField('Words:', validators=[validators.required()])



@app.route('/',methods=['GET'])
def index():
    form = ReusableForm(request.form)
    print(form.errors)
    return render_template('index.html',form=form)

def preapare_Response(message, status, result):
    data = {'status': status,
            'message': message}
    if result is not None:
        data['result'] = result

    res = make_response(jsonify(data))
    return res

@app.route("/predict",methods=['POST'])
def predict():
    if not request.is_json:
        return preapare_Response('Invalid data', 406, None)

    jsonobj = request.get_json()
    print(jsonobj)

    if jsonobj is None or "text" not in jsonobj or jsonobj.get("text") is None or len(jsonobj.get("text")) == 0:
        return preapare_Response('Invalid data', 400, None)

    labels = ['APPLICATION', 'BILL', 'BILL BINDER', 'BINDER', 'CANCELLATION NOTICE', 'CHANGE ENDORSEMENT',
              'DECLARATION',
              'DELETION OF INTEREST', 'EXPIRATION NOTICE', 'INTENT TO CANCEL NOTICE', 'NON-RENEWAL NOTICE',
              'POLICY CHANGE',
              'REINSTATEMENT NOTICE', 'RETURNED CHECK']

    # Take data value and get features
    words = jsonobj['text']
    #print(words)
    words=pd.DataFrame([words])
    words.columns = ['document_text']
    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 58627
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 500
    # This is fixed.
    EMBEDDING_DIM = 100

    X = token.texts_to_sequences(words['document_text'].values)

    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH,truncating='post', padding='post')

    print(model.summary())

    prediction = model.predict(X)
    print(prediction)
    maximum = np.amax(prediction, axis=1)
    index_of_maximum = prediction.argmax(axis=1)
    print(index_of_maximum)
    print(labels[index_of_maximum[0]])
    res = {'label' : labels[index_of_maximum[0]], 'confidence' : str(prediction[0, index_of_maximum[0]])}
    return preapare_Response(None, 200, res)








if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(host='0.0.0.0', port=50000, threaded=True)