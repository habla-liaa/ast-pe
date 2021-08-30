import tensorflow as tf
from tensorflow.python.keras import backend
import joblib
from pathlib import Path

def output_as_loss(y_true,y_pred):
    return tf.identity(y_pred)

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self,weights=None):
        super(WeightedBinaryCrossEntropy, self).__init__(name='weighted_binary_crossentropy')
        if isinstance(weights,str):
            weights = joblib.load(Path(weights).expanduser())
        self.weights = tf.constant(weights,dtype=tf.float32)

    def call(self,y_true,y_pred):
        return tf.reduce_mean(self.weights*backend.binary_crossentropy(y_true,y_pred),axis=-1)