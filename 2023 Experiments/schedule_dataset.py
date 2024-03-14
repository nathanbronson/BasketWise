import tensorflow as tf
from json import load
from sklearn.model_selection import train_test_split
import numpy as np
from random import sample

VALIDATION_PROP = .09
USE_YEARS = False
MAX_LEN = 440#1000
ITEMS = 1000

with open("./scheduledata.json", "r") as doc:
    __dd = load(doc)
with open("./scheduledata_lookups.json", "r") as doc:
    lk = load(doc)

p_types = len(list(lk["plays"])) + 1
s_types = len(list(lk["scores"])) + 1
#dd = [((d[0][0][:MAX_LEN], d[0][1][:MAX_LEN], d[0][2][:MAX_LEN], d[0][3]), d[1]) for d in dd]

seq_len = np.max([len(i[0]) for i in __dd["data"]])
stat_len = np.max([len(i[3][0]) for i in __dd["data"]])
sched_len = len(__dd["indices"][0][0])

t_len = len(__dd["data"])
__dd_ind, __vd_ind, __dd_labels, __vd_labels = train_test_split(__dd["indices"], __dd["labels"], test_size=VALIDATION_PROP)
__dd = {"indices": __dd_ind, "data": __dd["data"], "labels": __dd_labels}
__vd = {"indices": __vd_ind, "data": __dd["data"], "labels": __vd_labels}

def gen():
    dd = list(zip(__dd["indices"], __dd["labels"]))
    dd = sample(dd, len(dd))[:int(ITEMS)]
    while True:
        dd = sample(dd, len(dd))
        #fdd = [(([__dd["data"][game] for game in sched[0]], [__dd["data"][game] for game in sched[1]]), label) for sched, label in dd]
        #for item in fdd:
        for item in dd:
            sched = item[0]
            label = item[1]
            fdd = ([__dd["data"][game] for game in sched[0]], [__dd["data"][game] for game in sched[1]])
            #print(len([g[0] for g in fdd[0]]), len([g[0] for g in fdd[0]][0]))
            yield (
                (
                    (
                        tf.convert_to_tensor([g[0] for g in fdd[0]], dtype=tf.int32),
                        tf.convert_to_tensor([g[1] for g in fdd[0]], dtype=tf.int32),
                        tf.convert_to_tensor([g[2] for g in fdd[0]], dtype=tf.int32),
                        (
                            tf.convert_to_tensor([g[3][0] for g in fdd[0]], dtype=tf.float32),
                            tf.convert_to_tensor([g[3][1] for g in fdd[0]], dtype=tf.float32)
                        )
                    ),
                    (
                        tf.convert_to_tensor([g[0] for g in fdd[1]], dtype=tf.int32),
                        tf.convert_to_tensor([g[1] for g in fdd[1]], dtype=tf.int32),
                        tf.convert_to_tensor([g[2] for g in fdd[1]], dtype=tf.int32),
                        (
                            tf.convert_to_tensor([g[3][0] for g in fdd[1]], dtype=tf.float32),
                            tf.convert_to_tensor([g[3][1] for g in fdd[1]], dtype=tf.float32)
                        )
                    )
                ),
                tf.expand_dims(tf.convert_to_tensor(label, dtype=tf.int32), 0)
            )

def vgen():
    vd = list(zip(__vd["indices"], __vd["labels"]))
    while True:
        vd = sample(vd, len(vd))
        #fvd = [(([__vd["data"][game] for game in sched[0]], [__vd["data"][game] for game in sched[1]]), label) for sched, label in vd]
        for item in vd:
            sched = item[0]
            label = item[1]
            fvd = ([__vd["data"][game] for game in sched[0]], [__vd["data"][game] for game in sched[1]])
            yield (
                (
                    (
                        tf.convert_to_tensor([g[0] for g in fvd[0]], dtype=tf.int32),
                        tf.convert_to_tensor([g[1] for g in fvd[0]], dtype=tf.int32),
                        tf.convert_to_tensor([g[2] for g in fvd[0]], dtype=tf.int32),
                        (
                            tf.convert_to_tensor([g[3][0] for g in fvd[0]], dtype=tf.float32),
                            tf.convert_to_tensor([g[3][1] for g in fvd[0]], dtype=tf.float32)
                        )
                    ),
                    (
                        tf.convert_to_tensor([g[0] for g in fvd[1]], dtype=tf.int32),
                        tf.convert_to_tensor([g[1] for g in fvd[1]], dtype=tf.int32),
                        tf.convert_to_tensor([g[2] for g in fvd[1]], dtype=tf.int32),
                        (
                            tf.convert_to_tensor([g[3][0] for g in fvd[1]], dtype=tf.float32),
                            tf.convert_to_tensor([g[3][1] for g in fvd[1]], dtype=tf.float32)
                        )
                    )
                ),
                tf.expand_dims(tf.convert_to_tensor(label, dtype=tf.int32), 0)
            )

p_len = len(__dd["data"][0][0])
s_len = len(__dd["data"][0][3][0])

ds = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        (
            (
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                (
                    tf.TensorSpec(shape=(sched_len, stat_len,), dtype=tf.float32),
                    tf.TensorSpec(shape=(sched_len, stat_len,), dtype=tf.float32)
                )
            ),
            (
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                (
                    tf.TensorSpec(shape=(sched_len, stat_len,), dtype=tf.float32),
                    tf.TensorSpec(shape=(sched_len, stat_len,), dtype=tf.float32)
                )
            )
        ),
        tf.TensorSpec(shape=(1,), dtype=tf.int32)
    )
)#.batch(2).take(1).prefetch(20)
vds = tf.data.Dataset.from_generator(
    vgen,
    output_signature=(
        (
            (
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                (
                    tf.TensorSpec(shape=(sched_len, stat_len,), dtype=tf.float32),
                    tf.TensorSpec(shape=(sched_len, stat_len,), dtype=tf.float32)
                )
            ),
            (
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                tf.TensorSpec(shape=(sched_len, seq_len,), dtype=tf.int32),
                (
                    tf.TensorSpec(shape=(sched_len, stat_len,), dtype=tf.float32),
                    tf.TensorSpec(shape=(sched_len, stat_len,), dtype=tf.float32)
                )
            )
        ),
        tf.TensorSpec(shape=(1,), dtype=tf.int32)
    )
)

#print(list(ds.as_numpy_iterator())[0][0][0].shape)