import tensorflow as tf

class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps = 1000,decay=0.5,max_lr=0.001,multiplier=None):
        super(WarmupExponentialDecay, self).__init__()

        self.exponent = tf.cast(-decay,tf.float32)
        self.warmup_steps = tf.cast(warmup_steps,tf.float32)
        if max_lr and not multiplier:
            self.multiplier = max_lr/(warmup_steps**self.exponent)
        else:
            self.multiplier = multiplier
        self.multiplier = tf.cast(self.multiplier,tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step,tf.float32)
        arg1 = tf.math.pow(step,self.exponent)
        arg2 = step*tf.math.pow(self.warmup_steps,self.exponent-1.0)

        return self.multiplier * tf.math.minimum(arg1, arg2)

class PolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps = 1000,degree=1,max_lr=0.001,min_lr=0,num_updates=10000):
        super(PolynomialDecay, self).__init__()

        self.degree = tf.cast(degree,tf.float32)
        self.warmup_steps = tf.cast(warmup_steps,tf.float32)
        self.num_updates = tf.cast(num_updates,tf.float32)
        self.max_lr = tf.cast(max_lr,tf.float32)
        self.min_lr = tf.cast(min_lr,tf.float32)

        self.a = self.max_lr/tf.pow(self.warmup_steps,self.degree)
        self.b = (self.min_lr - self.max_lr)/(tf.pow(self.num_updates,self.degree) - tf.pow(self.warmup_steps,self.degree))
        self.c = (self.max_lr + self.min_lr - self.b*(tf.pow(self.num_updates, self.degree) + tf.pow(self.warmup_steps,self.degree)))/2

    def __call__(self, step):
        step = tf.cast(step,tf.float32)
        def ascend():
            return self.a*tf.pow(step,self.degree)
        def descend():
            return self.b*tf.pow(step,self.degree) + self.c
        return tf.cond(step <= self.warmup_steps,ascend,descend)

class DecayFactorCallback(tf.keras.callbacks.Callback):
    def __init__(self,variable_layer,variable_name,factor):
        super(DecayFactorCallback, self).__init__()
        self.factor=factor
        self.variable_layer = variable_layer
        self.variable_name = variable_name

    def on_train_begin(self, logs=None):
        variable_layer = [l for l in self.model.layers if l.name == self.variable_layer][0]
        self.updatable_variable = getattr(variable_layer,self.variable_name)
    
    def on_batch_end(self,batch,logs=None):
        old_val = tf.keras.backend.get_value(self.updatable_variable)
        tf.keras.backend.set_value(self.updatable_variable,old_val*self.factor)