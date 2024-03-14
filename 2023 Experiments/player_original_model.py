import tensorflow as tf
import numpy as np

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
        def __init__(self, d_model, dff, dropout_rate=0.1):
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

    class EncoderLayer(tf.keras.layers.Layer):
        def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
            super().__init__()

            self.self_attention = GlobalSelfAttention(
                num_heads=num_heads,
                key_dim=d_model,
                dropout=dropout_rate)

            self.ffn = FeedForward(d_model, dff)

        def call(self, x):
            x = self.self_attention(x)
            x = self.ffn(x)
            return x
    
    class PositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model
            self.reshape = tf.keras.layers.Reshape((-1, 1))
            self.embedding = tf.keras.layers.Dense(d_model)#tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
            #self.pos_encoding = positional_encoding(length=2048, depth=d_model)

        def compute_mask(self, *args, **kwargs):
            return self.embedding.compute_mask(*args, **kwargs)

        def call(self, x):
            x = self.reshape(x)
            x = self.embedding(x)
            # This factor sets the relative scale of the embedding and positonal_encoding.
            #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            #x = x + self.pos_encoding[tf.newaxis, :length, :]
            return x

    class Encoder(tf.keras.layers.Layer):
        def __init__(self, *, num_layers, d_model, num_heads,
                    dff, vocab_size, dropout_rate=0.1):
            super().__init__()

            self.d_model = d_model
            self.num_layers = num_layers

            self.pos_embedding = PositionalEmbedding(d_model=d_model)#PositionalEmbedding(
                #vocab_size=vocab_size, d_model=d_model)

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

            self.causal_self_attention = CausalSelfAttention(
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

            self.pos_embedding = DoublePositionalEmbedding(vocab_size=vocab_size,
                                                    d_model=d_model, max_game_length=4096, symmetric=True)
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
            self.dec_layers = [
                DecoderLayer(d_model=d_model, num_heads=num_heads,
                            dff=dff, dropout_rate=dropout_rate)
                for _ in range(num_layers)]

            self.last_attn_scores = None

        def call(self, x, context):
            # `x` is token-IDs shape (batch, target_seq_len)
            x = self.pos_embedding(*x)  # (batch_size, target_seq_len, d_model)

            x = self.dropout(x)

            for i in range(self.num_layers):
                x  = self.dec_layers[i](x, context)

            self.last_attn_scores = self.dec_layers[-1].last_attn_scores

            # The shape of x is (batch_size, target_seq_len, d_model).
            return x

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

            self.final_layer = tf.keras.layers.Dense(1, activation="sigmoid")

        def call(self, inputs):
            # To use a Keras model with `.fit` you must pass all your inputs in the
            # first argument.
            (plays, scores, times, (me_stats, op_stats)) = inputs
            #context, x  = inputs

            context = self.encoder(me_stats)  # (batch_size, context_len, d_model)

            x = self.decoder((plays, scores, times), context)  # (batch_size, target_len, d_model)

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

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)/50

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    BATCH_SIZE = 24
    CHECKPOINT_PATH = "./orig_timeseq"
    PREFETCH = tf.data.AUTOTUNE

    if __name__ == "__main__":
        from dataset import ds, vds, p_len, s_len, t_len, VALIDATION_PROP, p_types, s_types, lk
        ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(PREFETCH)
        vds = vds.batch(BATCH_SIZE, drop_remainder=True).prefetch(PREFETCH)
        model = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=None,
            target_vocab_size=(p_types, s_types),
            dropout_rate=dropout_rate
        )
        model.decoder.pos_embedding.set_embeddings(l=lk)#.positional_encoder.embedding.set_embeddings(l=lk)
        model(next(ds.take(1).as_numpy_iterator())[0])
        sched = CustomSchedule(d_model)
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
        #early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
        #if RESUME:
        #    if manager.restore_or_initialize() is not None:
        #        model.load_weights(CHECKPOINT_PATH)
        try:
            model.fit(ds, epochs=50, steps_per_epoch=int(t_len/BATCH_SIZE), callbacks=[cp_callback], validation_data=vds, validation_steps=int(VALIDATION_PROP/BATCH_SIZE * t_len))#, validation_data=vds)
        except Exception as err:
            manager.save()
            raise err
        manager.save()