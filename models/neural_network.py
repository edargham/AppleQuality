import tensorflow as tf
from keras import models, layers, activations


class NNmodel(models.Model):
    def __init__(
        self,
        num_hidden_layers: int=1,
        num_hidden_units: int=32,
        **kwargs
    ):
        super(NNmodel, self).__init__(**kwargs)
        self.model_layers = []

        for i in range(num_hidden_layers):
            self.model_layers.append(layers.Dense(units=num_hidden_units))
            self.model_layers.append(layers.LeakyReLU())

        self.model_layers.append(layers.Dense(1))

    def call(self, inputs, training=False):
        x = inputs

        for layer in self.model_layers:
            x = layer(x)

        return x
