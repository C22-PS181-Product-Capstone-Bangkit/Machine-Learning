import requests
import json
import pandas as pd
import numpy as np

restaurants = pd.read_csv('Main Dataset/Restaurant_list_new_ver.csv')


resp = requests.post('http://localhost:5000/predict',json={'user_ratings':[25]*18})


data = resp.json()

print("SORTED RESTO ID")
print(data['sorted_list'])
print("probabilities")
probs = data['id_to_probabilities']
rank = 1
print("id\trank\tprob")
for i in data['sorted_list']:
    print(i,'\t',rank,'\t',probs[i])
    rank += 1
