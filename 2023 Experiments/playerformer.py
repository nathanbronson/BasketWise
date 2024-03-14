import tensorflow as tf
import numpy as np

tf.random.set_seed(0)

with tf.device("/gpu:0"):

  def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

  class DoublePositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_game_length, symmetric=False):
      super().__init__()
      self.vocab_size = vocab_size
      self.symmetric = symmetric
      self.d_model = d_model
      #ADD CUSTOM SIGN EMBEDDING FOR SYMMETRY WITH JUST SIGNS AND VOCAB FROM STATS FILE
      self.plays_embedding = tf.keras.layers.Embedding(vocab_size[0], d_model, mask_zero=True) 
      self.scores_embedding = tf.keras.layers.Embedding(vocab_size[1], d_model, mask_zero=True) 
      self.pos_encoding = positional_encoding(length=max_game_length, depth=d_model)
      self.add = tf.keras.layers.Add()
      self.multiply = tf.keras.layers.Multiply()
      self.play_lookup = None
      self.score_lookup = None
      self.set_embeddings()
    
    def set_embeddings(self, pl=None, sl=None, l=None):
      #MAKE THIS IMPORTED
      if l is not None:
        pl = l["plays"]
        sl = l["scores"]
      if pl is not None:
        self.play_lookup = pl
      if sl is not None:
        self.score_lookup = sl
      if self.play_lookup is None:
        self.play_lookup = {"-30558": 1, "-20574": 2, "-20572": 3, "-20558": 4, "-20437": 5, "-20424": 6, "-618": 7, "-615": 8, "-607": 9, "-601": 10, "-598": 11, "-587": 12, "-586": 13, "-580": 14, "-579": 15, "-578": 16, "-574": 17, "-572": 18, "-558": 19, "-540": 20, "-521": 21, "-519": 22, "-449": 23, "-437": 24, "-412": 25, "-402": 26, "-97": 27, "0": 28, "97": 29, "402": 30, "412": 31, "437": 32, "449": 33, "519": 34, "521": 35, "540": 36, "558": 37, "572": 38, "574": 39, "578": 40, "579": 41, "580": 42, "586": 43, "587": 44, "598": 45, "601": 46, "607": 47, "615": 48, "618": 49, "20424": 50, "20437": 51, "20558": 52, "20572": 53, "20574": 54, "30558": 55}
      if self.score_lookup is None:
        self.score_lookup = {"-3": 1, "-2": 2, "-1": 3, "0": 4, "1": 5, "2": 6, "3": 7}
      self.play_lookup = {int(k): int(v) for k, v in self.play_lookup.items()}
      self.score_lookup = {int(k): int(v) for k, v in self.score_lookup.items()}
      self.play_sign_matrix = tf.convert_to_tensor([0] + [i/abs(i) if i != 0 else 1 for i in list(sorted(list(self.play_lookup.keys()), key=lambda e: self.play_lookup[e]))])
      self.score_sign_matrix = tf.convert_to_tensor([0] + [i/abs(i) if i != 0 else 1 for i in list(sorted(list(self.score_lookup.keys()), key=lambda e: self.score_lookup[e]))])
      self.play_abs_matrix = tf.convert_to_tensor([0] + [self.play_lookup[abs(i)] for i in list(sorted(list(self.play_lookup.keys()), key=lambda e: self.play_lookup[e]))])
      self.score_abs_matrix = tf.convert_to_tensor([0] + [self.score_lookup[abs(i)] for i in list(sorted(list(self.score_lookup.keys()), key=lambda e: self.score_lookup[e]))])

    def compute_mask(self, *args, **kwargs):
      return self.plays_embedding.compute_mask(*args, **kwargs)

    def call(self, plays, scores, times):
      #length = tf.shape(x)[1]
      #pos_enc = self.pos_encoding[tf.newaxis, times, :]
      plays = tf.cast(plays, tf.int32)
      scores = tf.cast(scores, tf.int32)
      times = tf.cast(times, tf.int32)
      pos_enc = tf.gather(self.pos_encoding, indices=times)
      embedded_plays = self.multiply([self.plays_embedding(tf.gather(self.play_abs_matrix, indices=plays)), tf.expand_dims(tf.cast(tf.gather(self.play_sign_matrix, indices=plays), tf.float32), 2)])
      embedded_scores = self.multiply([self.scores_embedding(tf.gather(self.score_abs_matrix, indices=scores)), tf.expand_dims(tf.cast(tf.gather(self.score_sign_matrix, indices=scores), tf.float32), 2)])
      x = self.add([embedded_plays, embedded_scores])
      # This factor sets the relative scale of the embedding and positonal_encoding.
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
      x = self.add([x, pos_enc])#this times index might not work
      return x

  class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
      super().__init__()
      self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
      self.layernorm = tf.keras.layers.LayerNormalization()
      self.add = tf.keras.layers.Add()

  class CrossAttention(BaseAttention):
    def call(self, x, context):
      attn_output, attn_scores = self.mha(
          query=x,
          key=context,
          value=context,
          return_attention_scores=True)

      # Cache the attention scores for plotting later.
      self.last_attn_scores = attn_scores

      x = self.add([x, attn_output])
      x = self.layernorm(x)

      return x

  class GlobalSelfAttention(BaseAttention):
    def call(self, x):
      attn_output = self.mha(
          query=x,
          value=x,
          key=x)
      x = self.add([x, attn_output])
      x = self.layernorm(x)
      return x

  class CausalSelfAttention(BaseAttention):
    def call(self, x):
      attn_output = self.mha(
          query=x,
          value=x,
          key=x,
          use_causal_mask = True)
      x = self.add([x, attn_output])
      x = self.layernorm(x)
      return x

  class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1, no_add=False):
      super().__init__()
      self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model),
        tf.keras.layers.Dropout(dropout_rate)
      ])
      self.add = tf.keras.layers.Add()
      self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
      x = self.add([x, self.seq(x)])
      x = self.layer_norm(x) 
      return x

  class NAFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1, no_add=False):
      super().__init__()
      self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model),
        tf.keras.layers.Dropout(dropout_rate)
      ])
      self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
      x = self.seq(x)
      x = self.layer_norm(x) 
      return x

  class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
      super().__init__()

      self.self_attention = GlobalSelfAttention(
          num_heads=num_heads,
          key_dim=d_model,
          dropout=dropout_rate)

      self.ffn = FeedForward(d_model, dff)

    def call(self, x):
      x = self.self_attention(x)
      #x = self.self_attention(tf.reshape(x, tf.shape(x)))#RESHAPE IS A REALLY WEIRD HACK
      x = self.ffn(x)
      return x

  class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                *,
                d_model,
                num_heads,
                dff,
                dropout_rate=0.1):
      super(DecoderLayer, self).__init__()

      self.causal_self_attention = GlobalSelfAttention(
          num_heads=num_heads,
          key_dim=d_model,
          dropout=dropout_rate)

      self.cross_attention = CrossAttention(
          num_heads=num_heads,
          key_dim=d_model,
          dropout=dropout_rate)

      self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
      x = self.causal_self_attention(x=x)
      x = self.cross_attention(x=x, context=context)

      # Cache the last attention scores for plotting later
      self.last_attn_scores = self.cross_attention.last_attn_scores

      x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
      return x

  class StatsCross(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                dropout_rate=0.1):
      super(StatsCross, self).__init__()

      self.d_model = d_model
      self.num_layers = num_layers

      self.dec_layers = [
          DecoderLayer(d_model=d_model, num_heads=num_heads,
                      dff=dff, dropout_rate=dropout_rate)
          for _ in range(num_layers)]

      self.last_attn_scores = None

    def call(self, x, context):
      for i in range(self.num_layers):
        x  = self.dec_layers[i](x, context)
      self.last_attn_scores = self.dec_layers[-1].last_attn_scores
      return x

  class PositionalEncoder(tf.keras.layers.Layer):
      def __init__(self, num_layers, d_model, num_heads,
                  dff, vocab_size, max_game_length, dropout_rate=0.1, symmetric=False):
          super().__init__()
          self.d_model = d_model
          self.num_layers = num_layers

          self.embedding = DoublePositionalEmbedding(vocab_size=vocab_size, d_model=d_model, max_game_length=max_game_length, symmetric=symmetric)

          self.enc_layers = [
              EncoderLayer(d_model=d_model,
                          num_heads=num_heads,
                          dff=dff,
                          dropout_rate=dropout_rate)
              for _ in range(num_layers)]
          self.dropout = tf.keras.layers.Dropout(dropout_rate)
      
      def call(self, plays, scores, times):
          x = self.embedding(plays, scores, times)
          x = self.dropout(x)
          for i in range(self.num_layers):
              x = self.enc_layers[i](x)
          return x

  class StatEncoder(tf.keras.layers.Layer):
      def __init__(self, num_layers, d_model, num_heads,
                  dff, num_stats_keys, dropout_rate=0.1):
          super().__init__()
          self.num_stats_keys = num_stats_keys
          self.d_model = d_model
          self.num_layers = num_layers

          self.stat_ff = NAFeedForward(d_model=d_model, dff=dff, no_add=True)

          self.enc_layers = [
              EncoderLayer(d_model=d_model,
                          num_heads=num_heads,
                          dff=dff,
                          dropout_rate=dropout_rate)
              for _ in range(num_layers)]
          #self.reshape = tf.keras.layers.Reshape((d_model, 1))
      
      #def compute_mask(self, x, mask=None):#NEEDS REIMPLEMENTED
      #  if mask is not None:
      #      return mask
      #  return tf.cast(tf.zeros_like(x), tf.bool)
      
      def call(self, stats):
          x = self.stat_ff(stats)#x = self.reshape(self.stat_ff(stats))
          for i in range(self.num_layers):
              x = self.enc_layers[i](x)
          return x
  
  class StatMask(tf.keras.layers.Layer):
    def __init__(self):
      super().__init__()
      self.supports_masking = True
    
    def compute_mask(self, x, mask=None):
      return tf.cast(tf.ones_like(x), tf.bool)
       
    def call(self, x):
      return x

  class FinalEncoder(tf.keras.layers.Layer):
      def __init__(self, num_layers, d_model, num_heads,
                  dff, dropout_rate=0.1):
          super().__init__()
          self.d_model = d_model
          self.num_layers = num_layers

          self.enc_layers = [
              EncoderLayer(d_model=d_model,
                          num_heads=num_heads,
                          dff=dff,
                          dropout_rate=dropout_rate)
              for _ in range(num_layers)]
      
      def call(self, x):
          for i in range(self.num_layers):
              x = self.enc_layers[i](x)
          return x

  #self.plays_embedding = PositionalEmbedding(num_play_types, d_model)#PositionalEncoder(num_layers=primary_layers, d_model=d_model, num_heads=primary_heads, dff=dff, vocab_size=num_play_types)#PositionalEmbedding(num_play_types, d_model)
  #self.scores_embedding = PositionalEmbedding(num_score_types, d_model)#PositionalEncoder(num_layers=primary_layers, d_model=d_model, num_heads=primary_heads, dff=dff, vocab_size=num_play_types)#PositionalEmbedding(num_score_types, d_model)
  class PlayerDense(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.concat = tf.keras.layers.Concatenate(axis=0)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=-np.inf),
            tf.keras.layers.Dense(40),
            tf.keras.layers.Dense(40, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(800, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
    
    def call(self, inputs):
        d1, d2 = inputs
        return self.model(d1 - d2)

  class PlayerFormer(tf.keras.Model):##SYMMETRIC EMBEDDINGS WONT WORK BECAUSE INDICES ARE STANDARDIZED
      def __init__(self, *, d_model, primary_heads=8, primary_layers=5, tertiary_heads=8, tertiary_layers=6, dff, num_stats_keys, dropout_rate=.1):
          super().__init__()
          self.mask = tf.keras.layers.Masking(mask_value=-np.inf)
          self.num_stats_keys = num_stats_keys
          self.stats_encoder = StatEncoder(primary_layers, d_model, primary_heads, dff, num_stats_keys, dropout_rate=dropout_rate)
          self.stats_cross = StatsCross(primary_layers, d_model, primary_heads, dff, dropout_rate=dropout_rate)
          self.me_stats_encoder = self.stats_encoder
          self.op_stats_encoder = self.stats_encoder
          self.me_stats_cross = self.stats_cross
          self.op_stats_cross = self.stats_cross
          self.subtract = tf.keras.layers.Subtract()
          self.final_representation = FinalEncoder(tertiary_layers, d_model, tertiary_heads, dff, dropout_rate=dropout_rate)
          self.flatten = tf.keras.layers.Flatten()
          self.project = tf.keras.layers.Dense(1, activation="sigmoid")
          self.stats_shape = tf.keras.layers.Reshape((-1, num_stats_keys))
      
      def call(self, inputs):
          me_roster, op_roster = inputs
          me_roster = self.mask(me_roster)
          op_roster = self.mask(op_roster)
          encoded_me = self.me_stats_encoder(self.stats_shape(me_roster))
          encoded_op = self.op_stats_encoder(self.stats_shape(op_roster))
          crossed_me = self.me_stats_cross(encoded_me, encoded_op)
          crossed_op = self.op_stats_cross(encoded_op, encoded_me)
          stats = self.subtract([crossed_me, crossed_op])
          x = self.final_representation(stats)
          x = self.flatten(x)
          x = self.project(x)
          return x

  @tf.function
  def rac(y_true, y_pred):
    return 1 - tf.math.reduce_mean(tf.math.abs(y_true - tf.math.round(y_pred)))

  D_MODEL = 128#32
  DFF = 256#64
  dropout_rate = 0.0#0.185
  BATCH_SIZE = 512#768
  PREFETCH = tf.data.AUTOTUNE
  CHECKPOINT_PATH = "./playerformer_checkpoint"
  RESUME = False

  class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
      super().__init__()

      self.d_model = d_model
      self.d_model = tf.cast(self.d_model, tf.float32)

      self.warmup_steps = warmup_steps
    
    def __call__(self, step):
      step = tf.cast(step, dtype=tf.float32)
      arg1 = tf.math.rsqrt(step)
      arg2 = step * (self.warmup_steps ** -1.5)

      return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

  if __name__ == "__main__":
    from player_dataset import ds, vds, stat_len, t_len, VALIDATION_PROP
    ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(PREFETCH)
    vds = vds.batch(BATCH_SIZE, drop_remainder=True).prefetch(PREFETCH)
    model = PlayerFormer(
      d_model=D_MODEL,
      dff=DFF,
      num_stats_keys=stat_len,
      dropout_rate=dropout_rate
    )
    #model = PlayerDense()
    #print(next(ds.take(1).as_numpy_iterator())[1])
    model(next(ds.take(1).as_numpy_iterator())[0])
    sched = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.legacy.Adam()#tf.keras.optimizers.legacy.Adam(sched, beta_1=0.9, beta_2=0.98, epsilon=1e-9)#tf.keras.optimizers.Adam()
    model.compile(
      loss="binary_crossentropy",
      optimizer=optimizer,
      metrics=["accuracy"]
    )
    print(model.summary())
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=CHECKPOINT_PATH, max_to_keep=2)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=120, restore_best_weights=True)
    if RESUME:
      if manager.restore_or_initialize() is not None:
        model.load_weights(CHECKPOINT_PATH)
    try:
        model.fit(ds, epochs=1000, steps_per_epoch=int(t_len/BATCH_SIZE), callbacks=[cp_callback, early_stop], validation_data=vds, validation_steps=int(VALIDATION_PROP/BATCH_SIZE * t_len))#, validation_data=vds)
    except Exception as err:
      manager.save()
      raise err
    print(model(next(ds.take(1).as_numpy_iterator())[0]))
    #print(model(next(ds.take(64).as_numpy_iterator())[0]))
    manager.save()
    #COULD EVENTUALLY TRY AN ACTIVATION THAT IS A DIMINISHING SINE (sin(x)e^x)