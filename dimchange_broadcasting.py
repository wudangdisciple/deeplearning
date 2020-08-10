import tensorflow as tf
import numpy as np
from tensorflow.keras import layers,Sequential,optimizers,datasets

#Reshape
a = tf.random.normal([4,28,28,3])
print(a.shape)
print(tf.reshape(a,[4,784,3]).shape)
print(tf.reshape(a,[4,-1,3]).shape)
print(tf.reshape(a,[4,784*3]).shape)
print(tf.reshape(a,[4,-1]).shape)
print(tf.reshape(tf.reshape(a,[4,-1]),[4,28,28,3]).shape)
print(tf.reshape(tf.reshape(a,[4,-1]),[4,14,56,3]).shape)
print(tf.reshape(tf.reshape(a,[4,-1]),[4,1,784,3]).shape)
#tf.transpose
a = tf.random.normal((4,3,2,1))
print(a.shape)
print(tf.transpose(a).shape)
print(tf.transpose(a,perm=[0,1,3,2]).shape)#4,3,1,2
a = tf.random.normal([4,28,28,3])
print(tf.transpose(a,[0,2,1,3]).shape)
print(tf.transpose(a,[0,3,2,1]).shape)
print(tf.transpose(a,[0,3,1,2]).shape)
#Expand dim
a = tf.random.normal([4,35,8])
print(tf.expand_dims(a,axis=0).shape)
print(tf.expand_dims(a,axis=3).shape)
print(tf.expand_dims(a,axis=-1).shape)
print(tf.expand_dims(a,axis=-4).shape)
#Squeeze dim
print(tf.squeeze(tf.zeros([1,2,1,1,3])).shape)
a = tf.zeros([1,2,1,3])
print(tf.squeeze(a,axis=0).shape)
print(tf.squeeze(a,axis=2).shape)
print(tf.squeeze(a,axis=-2).shape)
print(tf.squeeze(a,axis=-4).shape)
#Broadcasting
x = tf.random.normal([4,32,32,3])
print((x+tf.random.normal([3])).shape)
print((x+tf.random.normal([32,32,1])).shape)
print((x+tf.random.normal([4,1,1,1])).shape)
# print((x+tf.random.normal([1,4,1,1])).shape)#报错4不能广播为32
b = tf.broadcast_to(tf.random.normal([4,1,1,1]),[4,32,32,3])
print(b.shape)
#Broadcast VS Tile
a = tf.ones([3,4])
a1 = tf.broadcast_to(a,[2,3,4])
print(a1)
a2 = tf.expand_dims(a,axis=0)
print(a2)
a2 = tf.tile(a2,[2,1,1])
print(a2)