import tensorflow as tf
import numpy as np
from tensorflow.keras import layers,optimizers,datasets,Sequential

#from Numpy List
print(tf.convert_to_tensor(np.ones([2,3])))
print(tf.convert_to_tensor(np.zeros([2,3])))
print(tf.convert_to_tensor([1,2]))
print(tf.convert_to_tensor([1,2.]))
print(tf.convert_to_tensor([[1],[2.]]))
#tf.zeros
print(tf.zeros([]))
print(tf.zeros([1]))
print(tf.zeros([2,2]))
a = tf.zeros([2,3,3])
print(a)
#tf.zeros_like
print(tf.zeros_like(a))
print(tf.zeros(a.shape))
#tf.ones
print(tf.ones(1))
print(tf.ones([1]))
print(tf.ones([]))
print(tf.ones([2]))
print(tf.ones_like(a))
#Fill
print(tf.fill([2,2],0))
print(tf.fill([2,2],1))
#Normal
print(tf.random.normal([2,2],mean=1,stddev=1))
print(tf.random.normal([2,2]))
print(tf.random.truncated_normal([2,2],mean=0,stddev=1))
#uniform
print(tf.random.uniform([2,2],minval=0,maxval=1))
print(tf.random.uniform([2,2],minval=0,maxval=100))
#Random Permutation
idx = tf.range(10)
idx = tf.random.shuffle(idx)
print(idx)
aa = tf.random.normal([10,784])
bb = tf.random.uniform([10],maxval=10,dtype=tf.int32)
aa = tf.gather(aa,idx)
bb = tf.gather(bb,idx)
print(aa)
print(bb)
#tf.constant
print(tf.constant(1))
print(tf.constant([1]))
print(tf.constant([1,2.]))
#loss
out = tf.random.uniform([4,10])
print(out)
y = tf.range(4)
y = tf.one_hot(y,depth=10)
print(y)
loss = tf.keras.losses.mse(y,out)
print(loss)
mean_loss = tf.reduce_mean(loss)
print(mean_loss)
#matrix
x = tf.random.normal([4,784])
net = layers.Dense(10)#将784维转成10维
net.build((4,784))
print(net(x).shape)
print(net.kernel.shape)
print(net.bias.shape)
#Dim=4 Tensor
xx = tf.random.normal((4,32,32,3))
net = layers.Conv2D(16,kernel_size=3)
print(net(xx))