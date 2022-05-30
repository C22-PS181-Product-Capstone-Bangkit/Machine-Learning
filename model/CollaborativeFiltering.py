import tensorflow as tf
from tensorflow import keras


class CollaborativeFiltering(keras.Model):
    """
    A model of collaborative filtering using autoencoder. 
    """
    def __init__(self, encoder_layers = [10,5], decoder_layers = [5,10], latent_space_size = 5, **kwargs):
        super(CollaborativeFiltering,self).__init__(**kwargs)

        self.encoder = []
        for i in encoder_layers:
            self.encoder.append(keras.layers.Dense(i, activation='selu'))

        self.latent_space = keras.layers.Dense(latent_space_size, activation='selu')

        self.decoder = []
        for i in decoder_layers:
            self.decoder.append(keras.layers.Dense(i, activation='selu'))

    def call(self, inputs):
        x = keras.layers.Input(shape=(inputs.shape[1],))(inputs)
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        x = self.latent_space(x)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
        x = keras.layers.Dense(inputs.shape[1], activation='selu')(x)
        return x
            
