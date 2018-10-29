  def position_embedding(self,ret):
      pe = np.zeros(shape=[100,128])
      position = np.expand_dims(np.arange(0,100),-1)
      div_term = np.arange(0, 128, 2) * (-math.exp(math.log(10000.0) / 128))
      sin_ = np.sin(position * div_term)
      cos_ = np.cos(position * div_term)
      pe[:,0::2]=sin_
      pe[:,1::2]=cos_
      pe = tf.expand_dims(tf.constant(value=pe),0)
      pe = pe[:, :tf.shape(ret)[1]]
      pe = tf.cast(tf.expand_dims(pe, -2), tf.float32)
      ret = ret + pe
      return ret
