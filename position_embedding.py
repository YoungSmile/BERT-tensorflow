  # edit the SymbolModality of https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/modalities.py
  
  def position_embedding(self):
      pe = np.zeros(shape=[100,128])
      position = np.expand_dims(np.arange(0,100),-1)
      div_term = np.arange(0, 128, 2) * (-math.exp(math.log(10000.0) / 128))
      sin_ = np.sin(position * div_term)
      cos_ = np.cos(position * div_term)
      pe[:,0::2]=sin_
      pe[:,1::2]=cos_
      temp = tf.expand_dims(tf.constant(value=pe),0)
      return temp

  def bottom_simple(self, x, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
      # Ensure the inputs are 3-D
      if len(x.get_shape()) == 4:
        x = tf.squeeze(x, axis=3)
      while len(x.get_shape()) < 3:
        x = tf.expand_dims(x, axis=-1)



      var = self._get_weights()
      x = common_layers.dropout_no_scaling(
          x, 1.0 - self._model_hparams.symbol_dropout)
      ret = common_layers.gather(var, x)
      if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
        ret *= self._body_input_depth**0.5
      ret *= tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
      
### new
      pe=self.position_embedding()
      l = tf.shape(x)
      pe = pe[:,:l[1]]
      pe = tf.cast(tf.expand_dims(pe,-2),tf.float32)
      ret = ret + pe
###

      return ret
