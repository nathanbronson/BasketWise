import tensorflow as tf
from json import load
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
from random import sample

tf.random.set_seed(0)

VALIDATION_PROP = .09
USE_YEARS = False
MAX_LEN = 440#1000
SEQUENCE_T_RANDOM_F = True
PCT_INCREMENTS = 12

if USE_YEARS:
    dd = []
    for file in glob("./windata_*.json"):
        with open(file, "r") as doc:
            dd += load(doc)
else:
    with open("./windata.json", "r") as doc:
        dd = load(doc)
    with open("./windata_lookups.json", "r") as doc:
        lk = load(doc)

p_types = len(list(lk["plays"])) + 1
s_types = len(list(lk["scores"])) + 1
dd = [((d[0][0][:MAX_LEN], d[0][1][:MAX_LEN], d[0][2][:MAX_LEN], d[0][3]), d[1]) for d in dd]

seq_len = np.max([len(i[0][0]) for i in dd])
stat_len = np.max([len(i[0][3][0]) for i in dd])

t_len = len(dd)
__dd, __vd = train_test_split(dd, test_size=VALIDATION_PROP)
"""
@tf.function(input_signature=(
        (
            tf.TensorSpec(shape=(509,), dtype=tf.int32),
            tf.TensorSpec(shape=(509,), dtype=tf.int32),
            tf.TensorSpec(shape=(509,), dtype=tf.int32),
            (
                tf.TensorSpec(shape=(43,), dtype=tf.float32),
                tf.TensorSpec(shape=(43,), dtype=tf.float32)
            )
        ),
        tf.TensorSpec(shape=(1,), dtype=tf.int32)
    )
)
"""
@tf.function
def random_percent_mask(data):
    x, y = data
    #print(x)
    #tf.print(x, output_stream=sys.stdout)
    spot = tf.cast(tf.cast(tf.shape(x[0])[-1], tf.float32) * tf.random.uniform(shape=()), tf.int32)
    if tf.math.greater_equal(tf.random.uniform(shape=()), .9):
        return ((x[0], x[1], x[2], x[3]), y)
    mask = tf.concat([tf.ones_like(x[0])[:spot], tf.zeros_like(x[0])[spot:]], axis=0)
    mask = tf.convert_to_tensor(mask)
    return ((x[0] * mask, x[1] * mask, x[2] * mask, x[3]), y)

if SEQUENCE_T_RANDOM_F:
    @tf.function
    def percent_mask(data, pct):
        x, y = data
        #print(x)
        #tf.print(x, output_stream=sys.stdout)
        spot = tf.cast(tf.cast(tf.shape(x[0])[-1], tf.float32) * pct, tf.int32)
        mask = tf.concat([tf.ones_like(x[0])[:spot], tf.zeros_like(x[0])[spot:]], axis=0)
        mask = tf.convert_to_tensor(mask)
        return ((x[0] * mask, x[1] * mask, x[2] * mask, x[3]), y)
    
    def gen():
        dd = __dd
        for _ in range(2):
            dd = sample(dd, len(dd))
            for item in dd:
                yield random_percent_mask(
                    (
                        (
                            tf.convert_to_tensor(item[0][0], dtype=tf.int32),
                            tf.convert_to_tensor(item[0][1], dtype=tf.int32),
                            tf.convert_to_tensor(item[0][2], dtype=tf.int32),
                            (
                                tf.convert_to_tensor(item[0][3][0], dtype=tf.float32),
                                tf.convert_to_tensor(item[0][3][1], dtype=tf.float32)
                            )
                        ),
                        tf.expand_dims(tf.convert_to_tensor(item[-1], dtype=tf.int32), 0)
                    )
                )
        for i in range(PCT_INCREMENTS):
            pct = i/(PCT_INCREMENTS - 1)
            dd = sample(dd, len(dd))
            for item in dd:
                yield percent_mask(
                    (
                        (
                            tf.convert_to_tensor(item[0][0], dtype=tf.int32),
                            tf.convert_to_tensor(item[0][1], dtype=tf.int32),
                            tf.convert_to_tensor(item[0][2], dtype=tf.int32),
                            (
                                tf.convert_to_tensor(item[0][3][0], dtype=tf.float32),
                                tf.convert_to_tensor(item[0][3][1], dtype=tf.float32)
                            )
                        ),
                        tf.expand_dims(tf.convert_to_tensor(item[-1], dtype=tf.int32), 0)
                    ),
                    pct
                )
        while True:
            dd = sample(dd, len(dd))
            for item in dd:
                yield random_percent_mask(
                    (
                        (
                            tf.convert_to_tensor(item[0][0], dtype=tf.int32),
                            tf.convert_to_tensor(item[0][1], dtype=tf.int32),
                            tf.convert_to_tensor(item[0][2], dtype=tf.int32),
                            (
                                tf.convert_to_tensor(item[0][3][0], dtype=tf.float32),
                                tf.convert_to_tensor(item[0][3][1], dtype=tf.float32)
                            )
                        ),
                        tf.expand_dims(tf.convert_to_tensor(item[-1], dtype=tf.int32), 0)
                    )
                )
else:
    def gen():
        dd = __dd
        while True:
            dd = sample(dd, len(dd))
            for item in dd:
                yield random_percent_mask(
                    (
                        (
                            tf.convert_to_tensor(item[0][0], dtype=tf.int32),
                            tf.convert_to_tensor(item[0][1], dtype=tf.int32),
                            tf.convert_to_tensor(item[0][2], dtype=tf.int32),
                            (
                                tf.convert_to_tensor(item[0][3][0], dtype=tf.float32),
                                tf.convert_to_tensor(item[0][3][1], dtype=tf.float32)
                            )
                        ),
                        tf.expand_dims(tf.convert_to_tensor(item[-1], dtype=tf.int32), 0)
                    )
                )

def vgen():
    vd = __vd
    while True:
        vd = sample(vd, len(vd))
        for item in vd:
            yield random_percent_mask(
                (
                    (
                        tf.convert_to_tensor(item[0][0], dtype=tf.int32),
                        tf.convert_to_tensor(item[0][1], dtype=tf.int32),
                        tf.convert_to_tensor(item[0][2], dtype=tf.int32),
                        (
                            tf.convert_to_tensor(item[0][3][0], dtype=tf.float32),
                            tf.convert_to_tensor(item[0][3][1], dtype=tf.float32)
                        )
                    ),
                    tf.expand_dims(tf.convert_to_tensor(item[-1], dtype=tf.int32), 0)
                )
            )

p_len = len(dd[0][0][0])
s_len = len(dd[0][0][3][0])

ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        (
            tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            (
                tf.TensorSpec(shape=(stat_len,), dtype=tf.float32),
                tf.TensorSpec(shape=(stat_len,), dtype=tf.float32)
            )
        ),
        tf.TensorSpec(shape=(1,), dtype=tf.int32)
    )
)#.batch(2).take(1).prefetch(20)
vds = tf.data.Dataset.from_generator(
    vgen,
    output_signature=(
        (
            tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            (
                tf.TensorSpec(shape=(stat_len,), dtype=tf.float32),
                tf.TensorSpec(shape=(stat_len,), dtype=tf.float32)
            )
        ),
        tf.TensorSpec(shape=(1,), dtype=tf.int32)
    )
)

#print(list(ds.as_numpy_iterator())[0][0][0].shape)