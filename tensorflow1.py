import tensorflow as tf
import numpy as np
print(tf.convert_to_tensor(np.ones([2,3])))
print(tf.convert_to_tensor(np.zeros([2,3])))
print(tf.convert_to_tensor([1,2]))
print(tf.convert_to_tensor([1,2.]))
print(tf.convert_to_tensor([[1],[2.]]))#两行一列
print(tf.zeros([2,2]))
print(tf.random.normal([2,2],mean=1,stddev=1))
print(tf.random.normal([2,2]))
print(tf.random.truncated_normal([2,2],mean=0,stddev=1))#截断
print(tf.random.uniform([2,2],minval=0,maxval=1))#均匀采样
print(tf.random.uniform([10],maxval=1,dtype=tf.int32))
print(tf.random.shuffle([0,1,2,3,4,5]))
a = [4,35,8]
print(tf.gather(a,axis=0,indices=[2,1,3,0]))#axis为取哪个维度
a = tf.random.normal([4,28,28,3])#b h w c
print(tf.transpose(a,[0,3,2,1]))#输出b c w h
b = tf.random.normal([4,35,8])
print(tf.expand_dims(a,axis=0).shape)
print(tf.expand_dims(a,axis=3).shape)
print(tf.expand_dims(a,axis=-1).shape)
print(tf.expand_dims(a,axis=-4).shape)
print(tf.squeeze(tf.zeros([1,2,1,1,3])).shape)#删除为1的元素
print(tf.squeeze(tf.zeros([1,2,1,3]),axis=0).shape)

c = tf.ones([3,4])
print(c)
c1 = tf.broadcast_to(c,[2,3,4])
print(c1)
c2 = tf.expand_dims(c,axis=0)
c2 = tf.tile(c2,[2,1,1])#第一个元素复制2次 其余元素复制1次
print(c2)
