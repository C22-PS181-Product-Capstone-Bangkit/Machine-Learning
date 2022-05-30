import numpy as np
import pandas as pd
import csv

def preprocess_csv(csvfile : str, userid_field : str = 'userID', 
                   restoid_field : str = 'placeID', 
                   rating_field : str = 'total_rating'):
    """
    convert csv to user-resto rating matrix.
    """
    raw_data = pd.read_csv(csvfile)

    raw_data = raw_data.drop_duplicates([userid_field,restoid_field])

    # building the matrix
    user_resto_matrix = raw_data.pivot(index=userid_field,columns=restoid_field, values=rating_field)
    user_resto_matrix.fillna(0, inplace=True)
    users = user_resto_matrix.index.tolist()
    restos = user_resto_matrix.columns.tolist()

    return users, restos, user_resto_matrix.values






    