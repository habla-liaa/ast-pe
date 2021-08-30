import tensorflow as tf
import tensorflow.keras.layers as tfkl

from tensorflow.python.keras.engine import base_layer, training_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.eager import backprop

from tqdm import tqdm

class CustomModel(tf.keras.Model):
    def __init__(self, **kwargs):
        if 'ga_batches' in kwargs:
            self.ga_batches = tf.constant(kwargs.pop('ga_batches'),dtype=tf.int32)
        else:
            self.ga_batches = tf.constant(1,dtype=tf.int32)
        super(CustomModel, self).__init__(**kwargs)
        self.initial_step = 0
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        if self.ga_batches.numpy() > 1:
            self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

        if self._distribution_strategy is not None:
            self.num_replicas = self._distribution_strategy.num_replicas_in_sync
        else:
            self.num_replicas = 1

    #@tf.function
    def get_batch_gradients(self, data):
        if isinstance(data[0],list):
            x = [tf.constant(x_i) for x_i in data[0]]
        else:
            x = tf.constant(data[0])
        if isinstance(data[1],list):
            y_true = [tf.constant(y_i) for y_i in data[1]]
        else:
            y_true = tf.constant(data[1])

        self.n_acum_step.assign_add(1) #Track number of accumulation steps in GA

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y_true, y_pred,regularization_losses = self.losses)

        #self.compiled_metrics.update_state(y_true,y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)

        return loss, gradients, y_true, y_pred
    
    def train_step_ga_nofn(self,data):
        loss, gradients, y_true, y_pred = self.get_batch_gradients(data)
        #Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        tf.cond(tf.equal(self.n_acum_step, self.ga_batches), self.apply_accu_gradients, lambda: None)
        return loss, y_true, y_pred

    #@tf.function
    def train_step_ga(self,data):
        self.n_acum_step.assign_add(1) #Track number of accumulation steps in GA

        loss, gradients, y_true, y_pred = self.get_batch_gradients(data)
        #Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        tf.cond(tf.equal(self.n_acum_step, self.ga_batches), self.apply_accu_gradients, lambda: None)
        return loss, y_true, y_pred

    #@tf.function
    def train_step(self,data):
        loss, gradients, y_true, y_pred = self.get_batch_gradients(data)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss, y_true, y_pred

    def train_step_nofn(self,data):
        loss, gradients, y_true, y_pred = self.get_batch_gradients(data)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss, y_true, y_pred

    def apply_accu_gradients(self):
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))
    
    #@tf.function
    def distributed_input_from_generator(self, data):
        x, y = data
        if isinstance(x,list):
            x_distributed = list(zip([tf.split(x_i,num_or_size_splits = self.num_replicas) for x_i in x]))
        else:
            x_distributed = tf.split(x,self.num_replicas)
        if isinstance(y,list):
            y_distributed = list(zip([tf.split(y_i,num_or_size_splits = self.num_replicas) for x_i in x]))
        else:
            y_distributed = tf.split(y,num_or_size_splits = self.num_replicas)
        distributed_data = list(zip(x_distributed,y_distributed))

        def get_replica_batch(*args):
            args = tf.convert_to_tensor(args)
            replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
            return args[replica_id]

        return self._distribution_strategy.run(get_replica_batch, args=distributed_data)

    def distributed_input_from_generator_nofn(self, data):
        x, y = data
        if isinstance(x,list):
            x_distributed = list(zip([tf.split(x_i,num_or_size_splits = self.num_replicas) for x_i in x]))
        else:
            x_distributed = tf.split(x,self.num_replicas)
        if isinstance(y,list):
            y_distributed = list(zip([tf.split(y_i,num_or_size_splits = self.num_replicas) for x_i in x]))
        else:
            y_distributed = tf.split(y,num_or_size_splits = self.num_replicas)
        distributed_data = list(zip(x_distributed,y_distributed))

        def get_replica_batch(*args):
            args = tf.convert_to_tensor(args)
            replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
            return args[replica_id]

        return self._distribution_strategy.run(get_replica_batch, args=distributed_data)

    #@tf.function
    def distributed_train_step(self, data):
        distributed_data = self.distributed_input_from_generator_nofn(data)
        per_replica_losses, y_true, y_pred = self._distribution_strategy.run(self.train_step_nofn, args=(distributed_data,))

        return self._distribution_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None),y_true, y_pred

    def distributed_train_step_ga(self,data):
        distributed_data = self.distributed_input_from_generator(data)
        per_replica_losses,y_true, y_pred = self._distribution_strategy.run(self.train_step_ga_nofn, args=(distributed_data,))

        return self._distribution_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None),y_true, y_pred

    def non_distributed_train_step(self,data):
        return self.train_step(data)

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

        for cb in callbacks: cb.initial_step = self.initial_step
        
        with training_utils.RespectCompiledTrainableState(self), self.distributed_strategy.scope():
            for cb in callbacks: cb.model = self    #Make model accessible to callbacks
            for cb in callbacks: cb.on_train_begin(None)
            for epoch in range(initial_epoch, epochs):
                print('Epoch {}/{}'.format(epoch+1,epochs))
                for cb in callbacks: cb.on_epoch_begin(epoch,None)
                pbar = tqdm(range(len(data)),initial=self.initial_step)
                #pbar = tqdm.trange(self.initial_step, len(data), initial=self.initial_step)
                for step in pbar:
                    step = step + self.initial_step
                    for cb in callbacks: cb.on_train_batch_begin(step,None)
                    if self._distribution_strategy is not None:
                        if self.ga_batches.numpy() == 1:
                            loss, y_true, y_pred = self.distributed_train_step(data.__getitem__(step))
                        else:
                            loss, y_true, y_pred = self.distributed_train_step_ga(data.__getitem__(step))
                    else:
                        loss, y_true, y_pred = self.non_distributed_train_step(data.__getitem__(step))
                    
                    #self.reset_metrics()
                    #for metric in self.metrics: metric.update_state(y_true,y_pred)

                    pbar.set_description("Loss: {:.3f}".format(loss))
                    logs = {m.name: m.result() for m in self.metrics}

                    for cb in callbacks: cb.on_train_batch_end(step,logs)
                for cb in callbacks: cb.on_epoch_end(epoch, logs)
                self.initial_step = 0
            for cb in callbacks: cb.on_train_end(logs)
            return loss

        
