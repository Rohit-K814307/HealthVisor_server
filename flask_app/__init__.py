from flask import Flask
from flask_cors import CORS
from flask import Flask, jsonify
import requests
import tensorflow_text as text
from bert import tokenization
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
tf.gfile = tf.io.gfile
import tensorflow_hub as hub
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import re
import firebase_admin
from firebase_admin import db
import json


def basic_response():
    cred_obj = firebase_admin.credentials.Certificate('adminSDK.json')
    default_app = firebase_admin.initialize_app(cred_obj, {
        'databaseURL':"https://healthvisor-60926-default-rtdb.firebaseio.com/"
        })

    ref = db.reference("/")

    #sample query
    return json.loads(ref.order_by_child("steps").get())
    

def model(metrics):

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
        trainable=True)
    
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"] 
    sequence_output = outputs["sequence_output"]

    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(32, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(41, activation='softmax')(net)

    model = tf.keras.models.Model(inputs=text_input, outputs=out)
    model.compile(tf.keras.optimizers.legacy.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=metrics)
    
    return model

def encoded_mapping_func():
    encoded_mapping = {0: '(vertigo) Paroymsal  Positional Vertigo',
    1: 'AIDS',
    2: 'Acne',
    3: 'Alcoholic hepatitis',
    4: 'Allergy',
    5: 'Arthritis',
    6: 'Bronchial Asthma',
    7: 'Cervical spondylosis',
    8: 'Chicken pox',
    9: 'Chronic cholestasis',
    10: 'Common Cold',
    11: 'Dengue',
    12: 'Diabetes ',
    13: 'Dimorphic hemmorhoids(piles)',
    14: 'Drug Reaction',
    15: 'Fungal infection',
    16: 'GERD',
    17: 'Gastroenteritis',
    18: 'Heart attack',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    23: 'Hypertension ',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    26: 'Hypothyroidism',
    27: 'Impetigo',
    28: 'Jaundice',
    29: 'Malaria',
    30: 'Migraine',
    31: 'Osteoarthristis',
    32: 'Paralysis (brain hemorrhage)',
    33: 'Peptic ulcer diseae',
    34: 'Pneumonia',
    35: 'Psoriasis',
    36: 'Tuberculosis',
    37: 'Typhoid',
    38: 'Urinary tract infection',
    39: 'Varicose veins',
    40: 'hepatitis A'}

    return encoded_mapping

def clean_data(data):
    cleaned = []
 
    for line in data:
        line = line.lower() #makes it lowercase
 
        line = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-\\]", "", line) #takes out any symbols
 
        tokens = word_tokenize(line)
 
        words = [word for word in tokens if word.isalpha()] #check if only letters (no special chars/symbols)
 
        cleaned += words
 
    return cleaned


def new_pred(model,model_input,encoded_mapping):

    inp = clean_data(model_input)
    prediction = model.predict(inp)
    pred = np.array(prediction)

    #print(prediction)

    #print("_______")
    #print(pred)

    out = {}
    for i in range(len(pred)):
        max3 = np.argpartition(pred[i], -3)[-3:]

        for j in range(len(max3)):
            out["condition_"+str(j)] = encoded_mapping.get(max3[j])
            out["confidence_"+str(j)] = round((pred[i][max3[j]]) * 100)

    return out

def load_model(weights_dir):
    
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    bert_model = model(METRICS)
    bert_model.load_weights(weights_dir).expect_partial()

    return bert_model
    

def make_pred(inputVal,encoded_mapping,model):

    model_input = [inputVal]

    bert_model = model

    outVal = new_pred(bert_model,model_input,encoded_mapping)

    return outVal

def get_auth_code(client_id,client_secret):
    params = {
    'app_user_id': 'rohitk',
    'client_id': client_id,
    'client_secret': client_secret,
    }
    response = requests.post('https://api.1up.health/user-management/v1/user/auth-code', params=params).json()
    return response.get("code")

def get_access_code(client_id,client_secret,auth_code):
    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'code': auth_code,
        'grant_type': 'authorization_code',
    }
    response = requests.post('https://auth.1up.health/oauth2/token', data=data).json()
    return response.get("refresh_token"), response.get("access_token")

def refresh_access_code(client_id,client_secret,refresh_token):
    data = {
    'client_id': client_id,
    'client_secret': client_secret,
    'refresh_token': refresh_token,
    'grant_type': 'refresh_token',
    }
    response = requests.post('https://auth.1up.health/oauth2/token', data=data).json()
    return response.get("refresh_token"), response.get("access_token")

def get_total_data(access_token):
    url = "https://api.1up.health/fhir/dstu2/Practitioner?_public=true"
    payload={}
    headers = {
        'Authorization': 'Bearer ' + access_token
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    return response.json()

def query_data(city, state):

    client_id = "a4a19da1027bd432223f91d328c6b4b0"
    client_secret = "f2662a1b883f3020cef61d5b5c51fa7e"

    authorization = get_auth_code(client_id,client_secret)
    refresh_token, access_token = get_access_code(client_id,client_secret,authorization)
    data = get_total_data(access_token)

    #print("data found")
    
    responseValue = {"response":[]}

    for practitioner in data.get("entry"):

        #print('looping through practitioner')
        
        out = {}

        if 'resource' in list(practitioner.keys()):
            if 'address' in list(practitioner.get('resource').keys()):
                
                #print("address found")

                if 'state' in list(practitioner.get('resource').get('address')[0].keys()) and 'city' in list(practitioner.get('resource').get('address')[0].keys()):
                    
                    #print("state found")
                    
                    stateVal = practitioner.get('resource').get('address')[0].get('state')
                    cityVal = practitioner.get('resource').get('address')[0].get('city')

                    out['city'] = cityVal
                    out['state'] = stateVal

                if 'line' in list(practitioner.get('resource').get('address')[0].keys()):
                    
                    #print("line found")

                    roadAddress = practitioner.get('resource').get('address')[0].get('line')[0]

                    out['roadAdress'] = roadAddress

            if 'name' in list(practitioner.get('resource').keys()):
                

                #print("name found")
                
                if len(practitioner.get('resource').get('name').get("given")) > 1:

                    nameVal = str(practitioner.get('resource').get('name').get("given")[0]) + " " + str(practitioner.get('resource').get('name').get("given")[1]) + " " + str(practitioner.get('resource').get('name').get("family")[0])
                    
                else:
                    nameVal = str(practitioner.get('resource').get('name').get("given")) + " " + str(practitioner.get('resource').get('name').get("family")[0])
                

                

                if "suffix" in list(practitioner.get('resource').get('name').keys()):
                    nameVal += " " + practitioner.get('resource').get('name').get("suffix")[0]
                

                if "[" in nameVal:
                    nameVal = nameVal.replace("[", "")
                if "]" in nameVal:
                    nameVal = nameVal.replace("]", "")
                if "\'" in nameVal:
                    nameVal = nameVal.replace("\'","")
                
                out["name"] = nameVal


            if 'practitionerRole' in list(practitioner.get('resource').keys()):
                if 'display' in list(practitioner.get('resource').get('practitionerRole')[0].get('role').get('coding')[0].keys()):
                    

                    #print("role found")

                    role = practitioner.get('resource').get('practitionerRole')[0].get('role').get('coding')[0].get('display')
                    
                    out['role'] = role


        responseValue['response'].append(out)


    newResponse = {"response":[]}

    for entry in responseValue.get("response"):
        
        if city in entry.get('city') and state in entry.get('state'):
            
            #print(entry)
            newResponse['response'].append(entry)
        
        
    return newResponse

model = load_model("model_weights/bert_weights")
encoded_mapping = encoded_mapping_func()


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

app = Flask(__name__)
CORS(app)

"""

API ENDPOINTS BELOW

"""

#bert model
@app.route("/predict/<string:text>")
def modelpredict(text:str):

    description1 = text.replace("-"," ")
    responseVal = make_pred(description1,encoded_mapping,model)

    return jsonify({"prediction":responseVal})

################################################################################

#symptom steps
@app.route("/condition-steps/<string:inputVal>")
def conditionSteps(inputVal:str):

    datavals = basic_response()
    symptomSteps = datavals.get("steps")
    return jsonify({"steps":symptomSteps.get(inputVal)})

################################################################################

#query for the descriptions dataset
@app.route('/description-data/<string:condition>')
def descriptionData(condition:str):
    datavals = basic_response()
    descriptions = datavals.get("descriptions")

    return jsonify({"description":descriptions.get(condition)})

################################################################################

#query api to get doctor based on condition, role, city, and state
@app.route('/find-doctor/<string:city>/<string:state>')
def findDoctor(city:str,state:str):

    stateVal = state[0:-1] + state[-1].lower()

    return jsonify(query_data(city, stateVal))
