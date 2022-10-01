# import
from tensorflow.keras.layers import (
	Dense,
	Activation,
)
from tensorflow.keras.models import Model
from tensorflow.keras import Input

class NaiveClassifier:

	@staticmethod
	def build(dim, classes):
        inputShape = (dim, 1)

        # INPUT => Dense
        inputs = Input(inputShape)
        x = Dense(128)(inputs)
        x = Dense(128)(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="naive_classifier")

        return model

