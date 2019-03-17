from tensorflow.python.keras.layers import Embedding, Dense, Reshape, Concatenate
from tensorflow.python.keras import Input
import tensorflow as tf


sparse_input = [Input(shape=(1,),name='sparse_'+str(num)) for num in range(5)]
embed_layer= Embedding(30,8)

embed_list = [embed_layer(inp) for inp in sparse_input]

row = []
col = []
num_inputs = len(embed_list)

for i in range(num_inputs - 1):
    for j in range(i + 1, num_inputs):
        row.append(i)
        col.append(j)
p = tf.concat([embed_list[idx] for idx in row], axis=1)  # batch num_pairs k
q = tf.concat([embed_list[idx] for idx in col], axis=1)

inner_product = p * q
inner_product = tf.reduce_sum(inner_product, axis=2, keep_dims=True)
inner_product = tf.keras.layers.Flatten()(inner_product)

l1 = Concatenate(axis=-1)(embed_list)

linear_signal = tf.keras.layers.Reshape([len(embed_list)*8])(l1)

deep_input = tf.keras.layers.Concatenate()([linear_signal, inner_product])

print("ss")