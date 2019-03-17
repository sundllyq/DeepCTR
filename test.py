import keras
import tensorflow as tf
# input1 = keras.layers.Input(shape=(6,16,))
# x1 = keras.layers.Dense(9, activation='relu')(input1)
# input2 = keras.layers.Input(shape=(32,))
# x2 = keras.layers.Dense(9, activation='relu')(input2)
# # 相当于 added = keras.layers.add([x1, x2])
# added = keras.layers.Add()([x1, x2])
#
# out = keras.layers.Dense(4)(added)
# model = keras.models.Model(inputs=[input1, input2], outputs=out)
#
# print("S")

a = tf.constant([[[2,1],[2,1],[2,1],[2,1]],[[2,1],[2,1],[2,1],[2,1]],[[2,1],[2,1],[2,1],[2,1]]])
dd = tf.reduce_sum(a,axis=2,keep_dims=True)

# b = tf.constant([[3,5]])
#
# d = tf.constant([[[1]],[[2]]])
# e = tf.constant([[1],[1]])
#
# print(e.shape)
# print(d.shape)
#
# c = d+e
#
# aa = tf.constant([[[2],[2],[2],[2]],[[2],[2],[2],[2]],[[3],[3],[3],[3]]])
# cc = tf.constant([[2,3,4,1],[3,2,3,1],[2,5,1,3]])
#
# dd = tf.reduce_sum(aa,2)
#method2
with tf.Session() as sess:
    result2=sess.run(dd)
    print(result2)
    print(result2.shape)