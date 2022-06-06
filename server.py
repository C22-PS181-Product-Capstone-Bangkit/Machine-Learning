import numpy as np
import tensorflow as tf
import pandas as pd
import os
import warnings
from flask import Flask, request, jsonify

#app = Flask(__name__)

print('loading restaurant list')
restaurants = pd.read_csv('Main Dataset/Restaurant_list_new_ver.csv')
n_of_restos = restaurants.shape[0]
print("there are ",n_of_restos," restos")
print('loading model..')
model = tf.keras.models.load_model('./Main Dataset/modelv2')
print("testing the model")
test_pred = model.predict([np.arange(n_of_restos),np.random.randint(5,size=n_of_restos)])
test_pred = test_pred.reshape(-1)
test_pred_id_to_rank = (-test_pred).argsort()
print(test_pred_id_to_rank)
#print(restaurants.iloc[test_pred_id_to_rank])

print('starting the app')

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def compute_response():
    global model
    data = request.get_json(force=True)
    user_ratings = np.array(data['user_ratings'])
    pred = model.predict([np.arange(user_ratings.shape[0]),user_ratings])
    pred = pred.reshape(-1)
    #print("ok")
    #print(user_ratings)
    sorted_list = (-pred).argsort()
    #print(sorted_list)
    resp = {
        'sorted_list': [int(x) for x in sorted_list],
        'id_to_probabilities' : [float(x) for x in pred]
    }
    return jsonify(resp)

app.run(port=5000,debug=False)


