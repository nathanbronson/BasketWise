import tensorflow as tf
from json import load
from sklearn.model_selection import train_test_split
from random import sample
import numpy as np

VALIDATION_PROP = .09
sort_idx = 3
sort_key = lambda e: e[sort_idx]
MAX_LEN = 9
KILL_NON_MAX = True

with open("./playerdata.json", "r") as doc:
    __dd = load(doc)

_ad = []
if KILL_NON_MAX:
    for i in __dd:
        if len(list(filter(lambda e: sum(e) != 0, i[0][0]))) >= MAX_LEN and len(list(filter(lambda e: sum(e) != 0, i[0][1]))) >= MAX_LEN:
            _ad.append(i)
    __dd = _ad

#__dd = tf.keras.utils.pad_sequences(__dd, padding="post")
stat_len = len(__dd[0][0][0][0])
roster_len = MAX_LEN if MAX_LEN > 0 else len(__dd[0][0][0])

def _sort(l):#NOW SORTS
    _l = list(filter(lambda e: sum(e) != 0, l))
    pads = [[-np.inf for _ in range(stat_len)] for _ in range(len(l) - len(_l))]
    d = list(sorted(_l, key=sort_key)) + pads
    return d if MAX_LEN <= 0 else d[:MAX_LEN] #sample(l, len(l))

def shuff(l):
    return sample(l, len(l))

t_len = len(__dd)
__dd, __vd = train_test_split(__dd, test_size=VALIDATION_PROP)

def gen():
    dd = __dd
    while True:
        dd = shuff(dd)
        for item in dd:
            yield (
                (
                    tf.convert_to_tensor(_sort(item[0][0]), dtype=tf.float32),
                    tf.convert_to_tensor(_sort(item[0][1]), dtype=tf.float32),
                ),
                tf.expand_dims(tf.convert_to_tensor(item[-1], dtype=tf.int32), 0)
            )

def vgen():
    vd = __vd
    while True:
        vd = shuff(vd)
        for item in vd:
            yield (
                (
                    tf.convert_to_tensor(_sort(item[0][0]), dtype=tf.float32),
                    tf.convert_to_tensor(_sort(item[0][1]), dtype=tf.float32),
                ),
                tf.expand_dims(tf.convert_to_tensor(item[-1], dtype=tf.int32), 0)
            )

ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        (
            tf.TensorSpec(shape=(roster_len, stat_len), dtype=tf.float32),
            tf.TensorSpec(shape=(roster_len, stat_len), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(1,), dtype=tf.int32)
    )
)#.batch(2).take(1).prefetch(20)
vds = tf.data.Dataset.from_generator(
    vgen,
    output_signature=(
        (
            tf.TensorSpec(shape=(roster_len, stat_len), dtype=tf.float32),
            tf.TensorSpec(shape=(roster_len, stat_len), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(1,), dtype=tf.int32)
    )
)

#print(list(ds.as_numpy_iterator())[0][0][0].shape)