import tensorflow as tf
import tensorflow.keras.layers as tfkl

from tensorflow.python.keras.engine import base_layer, training_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.eager import backprop

from tqdm import tqdm

class CustomModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(CustomModel, self).__init__(**kwargs)


    @tf.function
    def train_step(self,data):
        x = data[0]
        y_true = data[1]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self._losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y_true,y_pred)

        return loss
    
    def fit(self,
            data,
            batch_size=None,
            epochs=1,
            verbose='auto',
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            initial_step=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):

        base_layer.keras_api_gauge.get_cell('fit').set(True)
        version_utils.disallow_legacy_graph('Model', 'fit')
        
        if callbacks is None:
            callbacks = []
        
        with training_utils.RespectCompiledTrainableState(self), self.distributed_strategy.scope():
            for cb in callbacks: cb.model = self    #Make model accessible to callbacks
            for cb in callbacks: cb.on_train_begin(None)
            for epoch in range(initial_epoch, epochs):
                print('Epoch {}/{}'.format(epoch+1,epochs))
                self.reset_metrics()
                for cb in callbacks: cb.on_epoch_begin(epoch,None)
                pbar = tqdm(range(initial_step, len(data)))
                for step in pbar:
                    for cb in callbacks: cb.on_train_batch_begin(step,None)
                    loss = self.train_step(data.__getitem__(step))

                    pbar.set_description("Loss: {:.3f}".format(loss))
                    logs = {m.name: m.result() for m in self.metrics}

                    for cb in callbacks: cb.on_train_batch_end(step,logs)
                for cb in callbacks: cb.on_epoch_end(epoch, logs)
            for cb in callbacks: cb.on_train_end(logs)
            return loss

        
