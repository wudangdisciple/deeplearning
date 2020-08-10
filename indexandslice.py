import tensorflow as tf
import numpy as np
from tensorflow.keras import layers,optimizers,datasets,Sequential
#Basic indexing
a = tf.ones([1,5,5,3])
print(a[0][0])
print(a[0][0][0])
print(a[0][0][0][2])
#Numpy-style indexing
a = tf.random.normal([4,28,28,3])
print(a[1].shape)
print(a[1,2].shape)
print(a[1,2,3].shape)
print(a[1,2,3,2].shape)
#...
a = tf.random.normal([2,4,28,28,3])
print(a[0].shape)
print(a[0,:,:,:,:].shape)
print(a[0,...].shape)
print(a[...,0].shape)
print(a[0,...,2].shape)
#tf.gather
a = tf.random.normal([4,35,8])#4个班级35个学生8门功课
print(tf.gather(a,axis=0,indices=[2,3]).shape)
print(a[2:4].shape)
print(tf.gather(a,axis=0,indices=[2,1,4,0]).shape)
print(tf.gather(a,axis=1,indices=[2,3,7,9,16]).shape)
print(tf.gather(a,axis=2,indices=[2,3,7]).shape)
#tf.gather_nd
print(tf.gather_nd(a,[0]).shape)#1个班级所有学生所有课程
print(tf.gather_nd(a,[0,1]).shape)#一个班级一个学生所有课程
print(tf.gather_nd(a,[0,1,2]).shape)#一个班级一个学生一门课程
print(tf.gather_nd(a,[[0,1,2]]).shape)
print(tf.gather_nd(a,[[0,0],[1,1]]).shape)#1班1号和2班2号的8门功课
print(tf.gather_nd(a,[[0,0,0],[1,1,1],[2,2,2]]).shape)#1班1号某1门和2班2号某一门和3班3号某一门功课
print(tf.gather_nd(a,[[[0,0,0],[1,1,1],[2,2,2]]]).shape)
#tf.boolean_mask
a = tf.random.normal([4,28,28,3])
print(tf.boolean_mask(a,mask=[True,True,False,False]).shape)
print(tf.boolean_mask(a,mask=[True,True,False],axis=3).shape)
a = tf.ones([2,3,4])
#0:0有4个 0:1无 0:2无 1:0无 1:1有4个 1:2有4个 所以总共长度为(2,4)
print(tf.boolean_mask(a,mask=[[True,False,False],[False,True,True]]))