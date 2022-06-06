from functools import lru_cache
from re import L
from tensorflow import keras
import tensorflow as tf
from preprocess import preprocess_csv
from model.CollaborativeFiltering import CollaborativeFiltering


# prepare data
users, restos, user_resto_matrix = preprocess_csv('data/rating_final.csv')
# put validation dataset here.. (will not use rating_final.csv, only as placeholder, the split is done in the csv)
val_users, val_restos, val_user_resto_matrix = preprocess_csv('data/rating_final.csv')

# create model

encoder_layers = [15,10]
latent_space = 5
decoder_layers = [10,15]
model = CollaborativeFiltering(encoder_layers=encoder_layers,
                               decoder_layers=decoder_layers,
                               latent_space_size=latent_space)
# defining loss function (MRMSE)
from tensorflow.keras import backend as KB
def m_rmse(gt, pred):
    mask = KB.cast(KB.not_equal(gt,0),KB.floatx())
    squared_error = KB.square(mask*(gt-pred))
    mrmse = KB.sqrt(KB.sum(squared_error)/KB.maximum(KB.sum(mask,axis=-1),1))
    return mrmse

# compile the model
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(learning_rate=0.001,momentum=0.9),loss=m_rmse,metrics=[m_rmse])
print(model.summary())
exit()

# train the model

# TODO: add callbacks

hist = model.fit(x = user_resto_matrix, y = user_resto_matrix,epochs=100, batch_size=35,validation_data=[user_resto_matrix,val_user_resto_matrix])

# TODO: add model testing here

# save the model
model_json = model.to_json()
with open('saved_model.json','w') as f:
    f.write(model_json)

model.save_weights('saved_model.h5')




