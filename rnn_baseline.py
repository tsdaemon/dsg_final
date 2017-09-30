import pandas as pd
import numpy as np
import keras
pd.options.display.max_columns = 999

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

df = pd.read_csv('../../dsg/demand_anonymized_20170802.csv',sep=';')

df["dateYear"] = df["First_MAD"].apply(lambda x: int(x[:4]))
df["dateMonth"] = df["First_MAD"].apply(lambda x: int(x[5:7]))
final_df = df.groupby(['SalOrg', 'Material', 'Month'])['OrderQty'].apply(lambda x: x.sum()).head(1).reset_index()

import itertools
test = pd.read_csv('../../dsg/demand_anonymized_20170802.csv',sep=';')
eval_comb = test[['Material', 'SalOrg']]
eval_comb = list(set([tuple(x) for x in eval_comb.values]))

comb = list(itertools.product(*[eval_comb, list(final_df['Month'].unique())]))
comb = [(t[0], t[1], m) for t, m in comb]

series2 = pd.DataFrame(comb, columns=['Material', 'SalOrg', 'Month'])
series2 = series2.sort_values(by=['Month', 'SalOrg', 'Material' ])
series2 = series2.merge(final_df, on=['Month', 'Material', 'SalOrg'], how='left')
series2 = series2.fillna(0)