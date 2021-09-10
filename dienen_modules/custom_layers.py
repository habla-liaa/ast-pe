import tensorflow as tf
import tensorflow.keras.layers as tfkl

class PEG(tfkl.Layer):
    def __init__(self, shape_2d, kernel_size=3, learn_cls_token=False, name=None, trainable=True):
        super().__init__(name=name, trainable=trainable)
        self.shape_2d = shape_2d
        self.kernel_size = kernel_size
        self.learn_cls_token = learn_cls_token

    def build(self,input_shape):
        self.conv_layer = tfkl.DepthwiseConv2D(self.kernel_size, padding='same')
        self.ch = input_shape[-1]
        self.bs = input_shape[0]
        self.spatial_reshape = tfkl.Reshape((self.shape_2d[0],self.shape_2d[1],self.ch))
        if self.learn_cls_token:
            self.cls_pe = self.add_weight(shape=(self.ch,),initializer='normal',trainable=True)

    def call(self,x):
        cls_embedding = tf.expand_dims(x[:,0],axis=1)
        feature_embeddings = x[:,1:]
        feature_spatial = self.spatial_reshape(feature_embeddings)
        pe = self.conv_layer(feature_spatial)
        feature_spatial += pe
        feature_embeddings_with_pe = tf.reshape(feature_spatial,tf.shape(feature_embeddings))
        if self.learn_cls_token:
            cls_embedding += self.cls_pe[tf.newaxis,tf.newaxis,:]

        return tf.concat([cls_embedding,feature_embeddings_with_pe],axis=1)
