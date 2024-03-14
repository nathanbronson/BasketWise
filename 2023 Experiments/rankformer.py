import tensorflow as tf
import numpy as np
from functools import partial

tf.random.set_seed(0)

with tf.device("/cpu:0"):

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

  class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_game_length):
      super().__init__()
      self.d_model = d_model
      self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
      self.pos_encoding = positional_encoding(length=max_game_length, depth=d_model)

    def compute_mask(self, *args, **kwargs):
      return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
      length = tf.shape(x)[1]
      x = self.embedding(x)
      # This factor sets the relative scale of the embedding and positonal_encoding.
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
      x = x + self.pos_encoding[tf.newaxis, :length, :]
      return x

  class DoublePositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_game_length, symmetric=False):
      super().__init__()
      self.vocab_size = vocab_size
      self.symmetric = symmetric
      self.d_model = d_model
      self.play_abs_embed = tf.keras.layers.Embedding(vocab_size[0], 1)
      self.play_sign_embedding = tf.keras.layers.Embedding(vocab_size[0], 1)
      self.score_abs_embed = tf.keras.layers.Embedding(vocab_size[1], 1)
      self.score_sign_embedding = tf.keras.layers.Embedding(vocab_size[1], 1)
      #ADD CUSTOM SIGN EMBEDDING FOR SYMMETRY WITH JUST SIGNS AND VOCAB FROM STATS FILE
      self.plays_embedding = tf.keras.layers.Embedding(vocab_size[0], d_model, mask_zero=True) 
      self.scores_embedding = tf.keras.layers.Embedding(vocab_size[1], d_model, mask_zero=True) 
      self.pos_encoding = positional_encoding(length=max_game_length, depth=d_model)
      self.add = tf.keras.layers.Add()
      self.abs = tf.keras.layers.Lambda(tf.math.abs if symmetric else tf.identity)
      self.sign = tf.keras.layers.Lambda(tf.math.sign if symmetric else tf.identity)
      self.multiply = tf.keras.layers.Multiply()
      self.reshape = tf.keras.layers.Reshape((-1, 1))
      self.dense_merge = False
      self.cat = tf.keras.layers.Concatenate(axis=-1)
      self.test_dense = tf.keras.layers.Dense(d_model)
      self.play_lookup = None
      self.score_lookup = None
      self.set_embeddings()
    
    def set_embeddings(self, pl=None, sl=None, l=None):
      """
      import json
      with open("./windata_lookups.json", "r") as doc:
        dat = json.load(doc)
      self.play_lookup = dat["plays"]
      self.score_lookup = dat["scores"]
      self.play_lookup = {int(k): int(v) for k, v in self.play_lookup.items()}
      self.score_lookup = {int(k): int(v) for k, v in self.score_lookup.items()}
      play_matrix = np.expand_dims(np.array([i/abs(i) if i != 0 else 1 for i in list(sorted(list(self.play_lookup.keys()), key=lambda e: self.play_lookup[e]))]), (0,))
      score_matrix = np.expand_dims(np.array([i/abs(i) if i != 0 else 1 for i in list(sorted(list(self.score_lookup.keys()), key=lambda e: self.score_lookup[e]))]), (0,))
      self.play_sign_embedding = tf.keras.layers.Embedding(self.vocab_size[0], 1, weights=[play_matrix], trainable=False, input_length=440)
      self.score_sign_embedding = tf.keras.layers.Embedding(self.vocab_size[1], 1, weights=[score_matrix], trainable=False, input_length=440)
      play_matrix = np.expand_dims(np.array([self.play_lookup[abs(i)] for i in list(sorted(list(self.play_lookup.keys()), key=lambda e: self.play_lookup[e]))]), (0,))
      score_matrix = np.expand_dims(np.array([self.score_lookup[abs(i)] for i in list(sorted(list(self.score_lookup.keys()), key=lambda e: self.score_lookup[e]))]), (0,))
      self.play_abs_embed = tf.keras.layers.Embedding(self.vocab_size[0], 1, weights=[play_matrix], trainable=False, input_length=440)
      self.score_abs_embed = tf.keras.layers.Embedding(self.vocab_size[1], 1, weights=[score_matrix], trainable=False, input_length=440)
      """
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
      #pos_enc = pos_enc[tf.newaxis, :]
      #print(tf.shape(pos_enc))
      if self.symmetric:
        embedded_plays = self.multiply([self.plays_embedding(tf.gather(self.play_abs_matrix, indices=plays)), tf.expand_dims(tf.cast(tf.gather(self.play_sign_matrix, indices=plays), tf.float32), 2)])
        embedded_scores = self.multiply([self.scores_embedding(tf.gather(self.score_abs_matrix, indices=scores)), tf.expand_dims(tf.cast(tf.gather(self.score_sign_matrix, indices=scores), tf.float32), 2)])
        #embedded_plays = self.multiply([self.plays_embedding(self.play_abs_embed(plays)), tf.cast(self.play_sign_embedding(plays), tf.float32)])
        #embedded_scores = self.multiply([self.scores_embedding(self.score_abs_embed(scores)), tf.cast(self.score_sign_embedding(scores), tf.float32)])
        #embedded_plays = self.multiply([self.plays_embedding(self.abs(plays)), tf.repeat(self.reshape(tf.cast(self.sign(plays), tf.float32)), repeats=self.d_model, axis=-1)])
        #embedded_scores = self.multiply([self.scores_embedding(self.abs(scores)), tf.repeat(self.reshape(tf.cast(self.sign(scores), tf.float32)), repeats=self.d_model, axis=-1)])
      else:
        embedded_plays = self.plays_embedding(plays)
        embedded_scores = self.scores_embedding(scores)
      if self.dense_merge:
        x = self.cat([embedded_plays, embedded_scores, pos_enc])
        x = self.test_dense(x)
      else:
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
      self.no_add = no_add
      self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model),
        tf.keras.layers.Dropout(dropout_rate)
      ])
      self.add = tf.keras.layers.Add()
      self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
      if self.no_add:
          x = self.seq(x)
      else:
          x = self.add([x, self.seq(x)])
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

  class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                dff, vocab_size, dropout_rate=0.1):
      super().__init__()

      self.d_model = d_model
      self.num_layers = num_layers

      self.pos_embedding = PositionalEmbedding(
          vocab_size=vocab_size, d_model=d_model)

      self.enc_layers = [
          EncoderLayer(d_model=d_model,
                      num_heads=num_heads,
                      dff=dff,
                      dropout_rate=dropout_rate)
          for _ in range(num_layers)]
      self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
      # `x` is token-IDs shape: (batch, seq_len)
      x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

      # Add dropout.
      x = self.dropout(x)

      for i in range(self.num_layers):
        x = self.enc_layers[i](x)

      return x  # Shape `(batch_size, seq_len, d_model)`.

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

  class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                dropout_rate=0.1):
      super(Decoder, self).__init__()

      self.d_model = d_model
      self.num_layers = num_layers

      self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                              d_model=d_model)
      self.dropout = tf.keras.layers.Dropout(dropout_rate)
      self.dec_layers = [
          DecoderLayer(d_model=d_model, num_heads=num_heads,
                      dff=dff, dropout_rate=dropout_rate)
          for _ in range(num_layers)]

      self.last_attn_scores = None

    def call(self, x, context):
      # `x` is token-IDs shape (batch, target_seq_len)
      x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

      x = self.dropout(x)

      for i in range(self.num_layers):
        x  = self.dec_layers[i](x, context)

      self.last_attn_scores = self.dec_layers[-1].last_attn_scores

      # The shape of x is (batch_size, target_seq_len, d_model).
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

          self.stat_ff = FeedForward(d_model=d_model, dff=dff, no_add=True)

          self.enc_layers = [
              EncoderLayer(d_model=d_model,
                          num_heads=num_heads,
                          dff=dff,
                          dropout_rate=dropout_rate)
              for _ in range(num_layers)]
          self.reshape = tf.keras.layers.Reshape((d_model, 1))
      
      #def compute_mask(self, x, mask=None):
      #  if mask is not None:
      #      return mask
      #  return tf.cast(tf.zeros_like(x), tf.bool)
      
      def call(self, stats):
          x = self.reshape(self.stat_ff(stats))
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

  class GameFormer(tf.keras.layers.Layer):##SYMMETRIC EMBEDDINGS WONT WORK BECAUSE INDICES ARE STANDARDIZED
      def __init__(self, *, num_play_types, d_model, num_score_types=len([-3, -2, -1, 0, 1, 2, 3]), primary_heads=4, primary_layers=1, secondary_heads=6, secondary_layers=2, tertiary_heads=8, tertiary_layers=4, dff, max_game_length=4096, num_stats_keys, output_d, dropout_rate=.1, enforced_symmetry=False, stats_override=True, use_stats=True):
          super().__init__()
          self.output_d = output_d
          self.supports_masking = True
          self.num_stats_keys = num_stats_keys
          self.stats_mask = StatMask()
          self.positional_encoder = PositionalEncoder(secondary_layers, d_model, secondary_heads, dff, (num_play_types, num_score_types), max_game_length, dropout_rate=dropout_rate, symmetric=enforced_symmetry)
          if use_stats:
            if enforced_symmetry or stats_override:
              self.stats_encoder = StatEncoder(primary_layers, d_model, primary_heads, dff, num_stats_keys, dropout_rate=dropout_rate)
              self.stats_cross = StatsCross(primary_layers, d_model, primary_heads, dff, dropout_rate=dropout_rate)
              self.me_stats_encoder = self.stats_encoder
              self.op_stats_encoder = self.stats_encoder
              self.me_stats_cross = self.stats_cross
              self.op_stats_cross = self.stats_cross
            else:
              self.me_stats_encoder = StatEncoder(primary_layers, d_model, primary_heads, dff, num_stats_keys, dropout_rate=dropout_rate)
              self.op_stats_encoder = StatEncoder(primary_layers, d_model, primary_heads, dff, num_stats_keys, dropout_rate=dropout_rate)
              self.me_stats_cross = StatsCross(primary_layers, d_model, primary_heads, dff, dropout_rate=dropout_rate)
              self.op_stats_cross = StatsCross(primary_layers, d_model, primary_heads, dff, dropout_rate=dropout_rate)
          self.subtract = tf.keras.layers.Subtract()
          if use_stats:
            self.game_stat_cross = StatsCross(primary_layers, d_model, primary_heads, dff, dropout_rate=dropout_rate)
            self.stat_game_cross = StatsCross(primary_layers, d_model, primary_heads, dff, dropout_rate=dropout_rate)
            self.join = tf.keras.layers.Concatenate(axis=-2)
          self.final_representation = FinalEncoder(tertiary_layers, d_model, tertiary_heads, dff, dropout_rate=dropout_rate)
          self.flatten = tf.keras.layers.Flatten()
          self.project = tf.keras.layers.Dense(output_d)
          self.stats_shape = tf.keras.layers.Reshape((num_stats_keys,))
          self.fix_shape = tf.keras.layers.Reshape((-1, d_model))
          self.gate_override = True
          self.use_stats = use_stats
        
      def compute_mask(self, inputs, mask=None):
        ref = inputs[-1][-1]
        nones = tf.expand_dims(tf.math.reduce_any(tf.cast(ref, tf.bool), axis=1), 1)
        return tf.repeat(nones, self.output_d, axis=1)
      
      def call(self, inputs):
          if tf.reduce_any(self.compute_mask(inputs)) or self.gate_override:
            plays, scores, times, (me_stats, op_stats) = inputs
            encoded_game_data = self.positional_encoder(plays, scores, times)
            if self.use_stats:
              encoded_me = self.me_stats_encoder(self.stats_shape(me_stats))
              encoded_op = self.op_stats_encoder(self.stats_shape(op_stats))
              crossed_me = self.me_stats_cross(encoded_me, encoded_op)
              crossed_op = self.op_stats_cross(encoded_op, encoded_me)
              stats = self.subtract([crossed_me, crossed_op])
              crossed_game = self.game_stat_cross(encoded_game_data, stats)
              crossed_stats = self.stat_game_cross(stats, encoded_game_data)
              joined_data = self.join([self.fix_shape(crossed_game), self.stats_mask(crossed_stats)])
            else:
              joined_data = encoded_game_data
            x = self.final_representation(joined_data)
            x = self.flatten(x)
            x = self.project(x)
          else:
            x = tf.random.normal((tf.shape(inputs[0])[0], self.output_d))
          return x

  class WinFormer(tf.keras.Model):
    def __init__(self, *, d_model, dff, num_score_types, num_play_types, num_stats_keys, output_d, dropout_rate=.1):
      super().__init__()
      self.game_former = GameFormer(
        num_play_types=num_play_types,
        d_model=d_model,
        num_score_types=num_score_types,
        dff=dff,
        num_stats_keys=num_stats_keys,
        output_d=output_d,
        dropout_rate=dropout_rate,
        enforced_symmetry=True,
        stats_override=True,
        use_stats=True
      )
      self.flatten = tf.keras.layers.Flatten()
      self.project = tf.keras.layers.Dense(1, activation="sigmoid")#try without sigmoid next
    
    def call(self, inputs):
      x = self.game_former(inputs)
      x = self.flatten(x)
      x = self.project(x)
      return x
    
  class RepresentationStack(tf.keras.layers.Layer):
    def __init__(self, *, d_model, dff, num_score_types, num_play_types, num_stats_keys, output_d, enforced_symmetry, stats_override, dropout_rate=.1):
      super().__init__()
      self.supports_masking = True
      self.game_former = GameFormer(
        num_play_types=num_play_types,
        d_model=d_model,
        num_score_types=num_score_types,
        dff=dff,
        num_stats_keys=num_stats_keys,
        output_d=output_d,
        dropout_rate=dropout_rate,
        enforced_symmetry=enforced_symmetry,
        stats_override=stats_override
      )
      self.time_dist = tf.keras.layers.TimeDistributed(self.game_former)
      self.flatten = tf.keras.layers.Flatten()
      self.project = tf.keras.layers.Dense(1, activation="sigmoid")
    
    def call(self, x):
      return self.time_dist(x)
    
  class BracketFormer(tf.keras.Model):
    def __init__(self, *, d_model, dff, num_score_types, num_play_types, num_stats_keys, output_d, cross_layers=2, cross_heads=6, self_layers=4, self_heads=8, dropout_rate=.1, enforced_symmetry=False, stats_override=True, subtract=True):
      super().__init__()
      self.rep_stack = RepresentationStack(
        num_play_types=num_play_types,
        d_model=d_model,
        num_score_types=num_score_types,
        dff=dff,
        num_stats_keys=num_stats_keys,
        output_d=output_d,
        dropout_rate=dropout_rate,
        enforced_symmetry=enforced_symmetry,
        stats_override=stats_override
      )
      if enforced_symmetry:
        self.stats_cross = StatsCross(cross_layers, d_model, cross_heads, dff, dropout_rate=dropout_rate)
        self.me_cross = self.stats_cross
        self.op_cross = self.stats_cross
      else:
        self.me_cross = StatsCross(cross_layers, d_model, cross_heads, dff, dropout_rate=dropout_rate)
        self.op_cross = StatsCross(cross_layers, d_model, cross_heads, dff, dropout_rate=dropout_rate)
      self.join = tf.keras.layers.Subtract() if subtract else tf.keras.layers.Concatenate(axis=-2)
      self.encoder = FinalEncoder(self_layers, d_model, self_heads, dff, dropout_rate=dropout_rate)
      self.flatten = tf.keras.layers.Flatten()
      self.project = tf.keras.layers.Dense(1, activation="sigmoid")
    
    def call(self, inputs):
      me, op = inputs
      me = self.rep_stack(me)
      op = self.rep_stack(op)
      crossed_me = self.me_cross(me, op)
      crossed_op = self.me_cross(op, me)
      crossed = self.join([crossed_me, crossed_op])
      encoded = self.encoder(crossed)
      x = self.flatten(encoded)
      x = self.project(x)
      return x
    """
    def train_step(self, data):#all winPCT masks
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    """

  class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                input_vocab_size, target_vocab_size, dropout_rate=0.1):
      super().__init__()
      self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate)

      self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=target_vocab_size,
                            dropout_rate=dropout_rate)

      self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
      # To use a Keras model with `.fit` you must pass all your inputs in the
      # first argument.
      context, x  = inputs

      context = self.encoder(context)  # (batch_size, context_len, d_model)

      x = self.decoder(x, context)  # (batch_size, target_len, d_model)

      # Final linear layer output.
      logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

      try:
        # Drop the keras mask, so it doesn't scale the losses/metrics.
        # b/250038731
        del logits._keras_mask
      except AttributeError:
        pass

      # Return the final output and the attention weights.
      return logits

  @tf.function
  def rac(y_true, y_pred):
    return 1 - tf.math.reduce_mean(tf.math.abs(y_true - tf.math.round(y_pred)))

  num_layers = 4
  D_MODEL = 32
  DFF = 64
  OUTPUT_D = 32
  num_heads = 8
  dropout_rate = 0.185
  BATCH_SIZE = 48#32
  PREFETCH = tf.data.AUTOTUNE
  CHECKPOINT_PATH = "./winformer_checkpoint"
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
    from dataset import ds, vds, p_len, s_len, t_len, VALIDATION_PROP, p_types, s_types, lk
    ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(PREFETCH)
    vds = vds.batch(BATCH_SIZE, drop_remainder=True).prefetch(PREFETCH)
    model = WinFormer(
      d_model=D_MODEL,
      dff=DFF,
      output_d=OUTPUT_D,
      num_score_types=s_types,
      num_play_types=p_types,
      num_stats_keys=s_len,
      dropout_rate=dropout_rate
    )
    model.game_former.positional_encoder.embedding.set_embeddings(l=lk)
    model(next(ds.take(1).as_numpy_iterator())[0])
    sched = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.legacy.Adam(sched, beta_1=0.9, beta_2=0.98, epsilon=1e-9)#tf.keras.optimizers.Adam()
    model.compile(
      loss="binary_crossentropy",
      optimizer=optimizer,
      metrics=["accuracy"]
    )
    print(model.summary())
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=CHECKPOINT_PATH, max_to_keep=2)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
    if RESUME:
      if manager.restore_or_initialize() is not None:
        model.load_weights(CHECKPOINT_PATH)
    try:
        model.fit(ds, epochs=50, steps_per_epoch=int(t_len/BATCH_SIZE), callbacks=[cp_callback, early_stop], validation_data=vds, validation_steps=int(VALIDATION_PROP/BATCH_SIZE * t_len))#, validation_data=vds)
    except Exception as err:
      manager.save()
      raise err
    manager.save()

  """
  transformer = Transformer(
      num_layers=num_layers,
      d_model=d_model,
      num_heads=num_heads,
      dff=dff,
      input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
      target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
      dropout_rate=dropout_rate)
  """
  """
  if __name__ == "__main__":#CURRENTLY OVERALLOCATES TO STATS
      g = GameFormer(num_play_types=20, d_model=64, dff=512, max_game_length=1024, num_stats_keys=100, output_d=128)
      lbi = g((
          tf.keras.layers.Input((300,), dtype=tf.int32),
          tf.keras.layers.Input((300,), dtype=tf.int32),
          tf.keras.layers.Input((300,), dtype=tf.int32),
          (tf.keras.layers.Input((100,), dtype=tf.float32),
          tf.keras.layers.Input((100,), dtype=tf.float32))
      ))
      gg = tf.function(g)
      from time import time
      times = []
      for i in range(8):#2**6=64 is optimal without function; 2**11 with
        d = (
            (10 * np.random.rand(2 ** (i + 1), 300)).astype("int32"),
            (6 * np.random.rand(2 ** (i + 1), 300)).astype("int32"),
            (10 * np.random.rand(2 ** (i + 1), 300)).astype("int32"),
            (np.random.rand(2 ** (i + 1), 100),
            np.random.rand(2 ** (i + 1), 100))
        )
        start = time()
        lbi = gg(d)
        times.append((time() - start)/(2 ** (i + 1)))
      print(*times, sep="\n")
      #l = g(((np.random.rand(300)*10).astype("int32"), (6 * np.random.rand(300)).astype("int32"),(10* np.random.rand(300)).astype("int32"), (np.random.rand(1, 100), np.random.rand(1, 100))))
      #print(tf.shape(l))
      #l_b = g(((np.random.rand(1, 128)*10).astype("int32"), (6 * np.random.rand(1, 128)).astype("int32"),(10* np.random.rand(1, 128)).astype("int32"), (np.random.rand(1, 100), np.random.rand(1, 100))))
      #l_b = g(((np.random.rand(1, 300)*10).astype("int32"), (6 * np.random.rand(1, 300)).astype("int32"),(10* np.random.rand(1, 300)).astype("int32"), (np.random.rand(1, 100), np.random.rand(1, 100))))
      #print(tf.shape(l_b))
  """
  """
  from json import load
    with open("./windata.json", "r") as doc:
        d = load(doc)
    y = np.array([_d[-1] for _d in d])
    #y = np.zeros((_y.shape[-1], 2))
    #y[:, _y] = 1
    x = [_d[0] for _d in d]
    play_index = len(np.unique(np.concatenate([np.unique(xd[0]) for xd in x])))
    score_index = len(np.unique(np.concatenate([np.unique(xd[1]) for xd in x])))
    stats_keys = len(x[0][-1][0])
    x = [(np.array(p), np.array(s), np.array(t), (np.array(s1), np.array(s2))) for p, s, t, (s1, s2) in x]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
    model = WinFormer(
      d_model=d_model,
      dff=dff,
      output_d=128,
      num_score_types=p_len,
      num_play_types=p_len,
      num_stats_keys=s_len
    )
    model((np.expand_dims(x[0][0], 0), np.expand_dims(x[0][1], 0), np.expand_dims(x[0][2], 0), (np.expand_dims(x[0][3][0], 0), np.expand_dims(x[0][3][1], 0))))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
      loss="binary_crossentropy",
      optimizer=optimizer,
      metrics=["accuracy"]
    )
    print(model.summary())
    #print(np.array(x_train)[0], np.array(x_train).shape, np.array(y_train)[0], np.array(y_train).shape)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=1)
  """